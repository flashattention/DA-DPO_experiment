import os
import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoProcessor, AutoModel# ==========================================
# [GPU 할당 제한] 
# A100 80GB 2대(0번, 1번)만 사용하여 VRAM을 여유롭게 잡고 통신 병목을 없앱니다.
# 반드시 다른 torch 라이브러리가 초기화되기 전, 스크립트 최상단에 선언해야 합니다.
# ==========================================
os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"  # A100 80GB 2대만 사용

# ==========================================
# [경로 자동 설정]
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

INPUT_DPO_PATH = os.path.join(DATA_DIR, "train_dpo_pairs.jsonl")
ORIGINAL_DATA_PATH = os.path.join(DATA_DIR, "merged_eval_data.jsonl")
OUTPUT_PATH = os.path.join(DATA_DIR, "train_hc_dpo_pairs.jsonl")

# ==========================================
# [설정] Reference 모델 세팅 (Qwen3-32B)
# ==========================================
# 우리가 최종적으로 파인튜닝할 32B 모델을 Reference로 사용합니다.
MODEL_ID = "Qwen/Qwen3-VL-32B-Thinking"

# 하이퍼파라미터 (논문 실험 시 조정 가능)
ALPHA = 2.0  # Logprob Margin의 영향력
BETA = 1.5   # Human Penalty의 영향력

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def calculate_response_logprob(model, processor, prompt_text, response_text, image_path):
    """
    Qwen-VL 공식 Chat Template을 적용하여 정밀하게 Logprob을 계산합니다.
    """
    # 1. Qwen-VL이 이해할 수 있는 메시지 리스트(Dict) 구조 생성
    messages_full = [
        {"role": "user", "content": [{"type": "image", "image": image_path}, {"type": "text", "text": prompt_text}]},
        {"role": "assistant", "content": [{"type": "text", "text": response_text}]}
    ]
    
    # 답변이 제외된 프롬프트까지만의 메시지
    messages_prompt = [
        {"role": "user", "content": [{"type": "image", "image": image_path}, {"type": "text", "text": prompt_text}]}
    ]

    # 2. 프로세서를 통해 텐서(Tensor)로 완벽하게 변환 (특수 토큰 자동 삽입)
    # add_generation_prompt=True를 하면 <|im_start|>assistant\n 까지 자동으로 붙여줍니다.
    inputs_full = processor(
        text=[processor.apply_chat_template(messages_full, tokenize=False)],
        images=[image_path],
        return_tensors="pt",
        padding=True
    ).to(model.device)
    
    inputs_prompt = processor(
        text=[processor.apply_chat_template(messages_prompt, tokenize=False, add_generation_prompt=True)],
        images=[image_path],
        return_tensors="pt"
    )

    # 3. 프롬프트 부분의 길이(Token 수) 계산
    # 이 길이를 알아야 '답변(Response)' 부분만의 Logprob을 잘라낼 수 있습니다.
    prompt_len = inputs_prompt["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model(**inputs_full)
        logits = outputs.logits  # [batch_size, seq_len, vocab_size]
        
    input_ids = inputs_full["input_ids"][0]
    
    # 4. 정답 토큰 매칭을 위해 Logits와 Labels를 한 칸씩 Shift
    shift_logits = logits[0, prompt_len-1:-1, :].contiguous()
    shift_labels = input_ids[prompt_len:].contiguous()
    
    # 5. Log Softmax 계산 및 타겟 토큰의 Logprob 추출
    log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
    token_log_probs = torch.gather(log_probs, 1, shift_labels.unsqueeze(1)).squeeze(-1)
    
    # 평균 Logprob 반환
    return token_log_probs.mean().item()

def main():
    print(f"🚀 Reference Model ({MODEL_ID}) 로딩 중... (GPU VRAM 확인 요망)")
    
    # A100 8대 환경이므로 device_map="auto"로 여러 GPU에 분산 적재
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        MODEL_ID, 
        device_map="auto", 
        dtype=torch.bfloat16,
        trust_remote_code=True
    )
    model.eval()
    
    print("✅ 모델 로딩 완료. 데이터 병합을 시작합니다.")
    
    # 1. 원본 데이터에서 Human Metadata(overall_match_rate) 추출을 위해 로드
    raw_data = []
    with open(ORIGINAL_DATA_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            raw_data.append(json.loads(line))
            
    # image_path를 기준으로 match_rate 매핑 딕셔너리 생성
    match_rate_dict = {
        item['metadata']['image_path']: item.get('eval_similarity', {}).get('overall_match_rate', 1.0)
        for item in raw_data
    }
    
    # 2. 1번 스크립트에서 만든 DPO Pair 로드
    dpo_pairs = []
    with open(INPUT_DPO_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            dpo_pairs.append(json.loads(line))
            
    hc_dpo_dataset = []
    
    print("🧠 HC-DA-DPO 난이도(Difficulty Weight) 계산 시작...")
    for item in tqdm(dpo_pairs, desc="Calculating Logprobs & Weights"):
        image_path = item["image_path"]
        prompt = item["prompt"]
        chosen = item["chosen"]
        rejected = item["rejected"]
        
        # 실제 이미지의 절대 경로 (필요시 경로 수정)
        full_image_path = os.path.join(PROJECT_ROOT, image_path) 
        
        try:
            # 1. Logprob Margin 계산
            logprob_chosen = calculate_response_logprob(model, processor, prompt, chosen, full_image_path)
            logprob_rejected = calculate_response_logprob(model, processor, prompt, rejected, full_image_path)
            
            margin = logprob_chosen - logprob_rejected
            
            # 2. Human Penalty 계산 (1.0 - match_rate)
            match_rate = match_rate_dict.get(image_path, 1.0)
            human_penalty = 1.0 - match_rate
            
            # 3. 최종 HC-DA-DPO Weight 산출 (수식 적용)
            # 마진이 작을수록(어려울수록), 인간 페널티가 클수록 가중치 상승
            weight_logit = (-ALPHA * margin) + (BETA * human_penalty)
            difficulty_weight = sigmoid(weight_logit)
            
            # 데이터에 가중치 및 디버깅용 정보 추가
            item["difficulty_weight"] = float(difficulty_weight)
            item["logprob_margin"] = float(margin)
            item["human_penalty"] = float(human_penalty)
            
            hc_dpo_dataset.append(item)
            
        except Exception as e:
            print(f"\n⚠️ Error processing {image_path}: {e}")
            continue

    # 3. 최종 데이터셋 저장
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for item in hc_dpo_dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    print(f"\n🎉 HC-DA-DPO 가중치가 포함된 데이터셋 저장 완료! -> {OUTPUT_PATH}")
    print(f"생성된 데이터 수: {len(hc_dpo_dataset)} 건")

if __name__ == "__main__":
    main()