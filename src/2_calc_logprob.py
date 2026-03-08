import os
import json
import torch
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from PIL import Image # 이미지 로딩을 위한 PIL 추가
from transformers import AutoProcessor, AutoModelForImageTextToText, AutoModel

# ==========================================
# [GPU 할당 제한] 
# A100 80GB 2대(0번, 1번)만 사용하여 VRAM을 여유롭게 잡고 통신 병목을 없앱니다.
# ==========================================
os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"

# ==========================================
# [경로 및 버전 자동 설정]
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# 🌟 1번 스크립트에서 만든 '최신(latest)' DPO 데이터를 우선 사용
INPUT_DPO_LATEST = os.path.join(DATA_DIR, "train_dpo_pairs_latest.jsonl")
INPUT_DPO_DEFAULT = os.path.join(DATA_DIR, "train_dpo_pairs.jsonl")
ORIGINAL_DATA_PATH = os.path.join(DATA_DIR, "merged_eval_data.jsonl")

# 🌟 [버전 관리] 현재 시간을 바탕으로 고유한 타임스탬프 생성
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_VERSIONED = os.path.join(DATA_DIR, f"train_hc_dpo_pairs_{TIMESTAMP}.jsonl")
OUTPUT_LATEST = os.path.join(DATA_DIR, "train_hc_dpo_pairs_latest.jsonl")

# ==========================================
# [설정] Reference 모델 세팅 (Qwen3-32B)
# ==========================================
MODEL_ID = "Qwen/Qwen3-VL-32B-Thinking"

ALPHA = 2.0  
BETA = 1.5   

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def resolve_image_path(image_path: str) -> str:
    """
    데이터셋의 상대 경로를 실제 파일 절대 경로로 변환.
    우선순위:
    1) 절대 경로 그대로
    2) PROJECT_ROOT 기준
    3) DATA_DIR 기준 (현재 데이터 구조: data/images/...)
    """
    if os.path.isabs(image_path):
        return image_path

    candidate_paths = [
        os.path.join(PROJECT_ROOT, image_path),
        os.path.join(DATA_DIR, image_path),
    ]

    for p in candidate_paths:
        if os.path.exists(p):
            return p

    # 디버깅/경고용으로 기본 후보 반환
    return candidate_paths[0]

def calculate_response_logprob(model, processor, prompt_text, response_text, full_image_path):
    """
    Qwen-VL 공식 Chat Template 적용. (Base64 파싱 버그 해결)
    """
    # 🛠️ [버그 픽스] PIL을 이용해 이미지를 직접 로드하여 Base64 디코딩 에러 원천 차단
    raw_image = Image.open(full_image_path).convert("RGB")
    file_uri = f"file://{full_image_path}" # Chat template 호환성을 위한 URI 포맷
    
    messages_full = [
        {"role": "user", "content": [{"type": "image", "image": file_uri}, {"type": "text", "text": prompt_text}]},
        {"role": "assistant", "content": [{"type": "text", "text": response_text}]}
    ]
    
    messages_prompt = [
        {"role": "user", "content": [{"type": "image", "image": file_uri}, {"type": "text", "text": prompt_text}]}
    ]

    # 🛠️ [버그 픽스] images 인자에 문자열 경로 대신 PIL Image 객체(raw_image) 전달
    inputs_full = processor(
        text=[processor.apply_chat_template(messages_full, tokenize=False)],
        images=[raw_image], 
        return_tensors="pt",
        padding=True
    ).to(model.device)
    
    inputs_prompt = processor(
        text=[processor.apply_chat_template(messages_prompt, tokenize=False, add_generation_prompt=True)],
        images=[raw_image],
        return_tensors="pt"
    )

    prompt_len = inputs_prompt["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model(**inputs_full)
        logits = outputs.logits 
        
    input_ids = inputs_full["input_ids"][0]
    
    shift_logits = logits[0, prompt_len-1:-1, :].contiguous()
    shift_labels = input_ids[prompt_len:].contiguous()
    
    log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
    token_log_probs = torch.gather(log_probs, 1, shift_labels.unsqueeze(1)).squeeze(-1)
    
    return token_log_probs.mean().item()

def main():
    print(f"🚀 Reference Model ({MODEL_ID}) 로딩 중... (GPU: 6, 7 사용)")
    
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    try:
        # Qwen3-VL은 Image-Text-Text 모델 클래스로 로드해야 가중치 mismatch 경고를 피할 수 있음
        model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID,
            device_map="auto",
            dtype=torch.bfloat16,
            trust_remote_code=True
        )
    except Exception as e:
        print(f"⚠️ AutoModelForImageTextToText 로드 실패. AutoModel로 폴백합니다: {e}")
        model = AutoModel.from_pretrained(
            MODEL_ID,
            device_map="auto",
            dtype=torch.bfloat16,
            trust_remote_code=True
        )
    model.eval()
    
    print("✅ 모델 로딩 완료. 데이터 병합을 시작합니다.")
    
    raw_data = []
    with open(ORIGINAL_DATA_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            raw_data.append(json.loads(line))
            
    match_rate_dict = {
        item['metadata']['image_path']: item.get('eval_similarity', {}).get('overall_match_rate', 1.0)
        for item in raw_data
    }
    
    input_dpo_path = INPUT_DPO_LATEST if os.path.exists(INPUT_DPO_LATEST) else INPUT_DPO_DEFAULT
    if not os.path.exists(input_dpo_path):
        raise FileNotFoundError(
            f"입력 DPO 파일이 없습니다. 확인한 경로: {INPUT_DPO_LATEST}, {INPUT_DPO_DEFAULT}"
        )

    dpo_pairs = []
    print(f"📂 입력 데이터 로드: {input_dpo_path}")
    with open(input_dpo_path, 'r', encoding='utf-8') as f:
        for line in f:
            dpo_pairs.append(json.loads(line))
            
    hc_dpo_dataset = []
    
    print("🧠 HC-DA-DPO 난이도(Difficulty Weight) 계산 시작...")
    for item in tqdm(dpo_pairs, desc="Calculating Logprobs & Weights"):
        image_path = item["image_path"]
        prompt = item["prompt"]
        chosen = item["chosen"]
        rejected = item["rejected"]
        
        full_image_path = resolve_image_path(image_path)
        
        # 이미지 파일 존재 여부 사전 체크 (안전 장치)
        if not os.path.exists(full_image_path):
            print(f"\n⚠️ 경고: 이미지 파일을 찾을 수 없습니다 -> {full_image_path}")
            continue
            
        try:
            logprob_chosen = calculate_response_logprob(model, processor, prompt, chosen, full_image_path)
            logprob_rejected = calculate_response_logprob(model, processor, prompt, rejected, full_image_path)
            
            margin = logprob_chosen - logprob_rejected
            match_rate = match_rate_dict.get(image_path, 1.0)
            human_penalty = 1.0 - match_rate
            
            weight_dpo = 1.0
            weight_da_dpo = sigmoid(-ALPHA * margin)
            weight_hc_da_dpo = sigmoid((-ALPHA * margin) + (BETA * human_penalty))
            
            item["weight_dpo"] = float(weight_dpo)
            item["weight_da_dpo"] = float(weight_da_dpo)
            item["weight_hc_da_dpo"] = float(weight_hc_da_dpo)
            item["logprob_margin"] = float(margin)
            item["human_penalty"] = float(human_penalty)
            
            hc_dpo_dataset.append(item)
            
        except Exception as e:
            print(f"\n⚠️ Error processing {image_path}: {e}")
            continue

    # 🌟 3. 최종 데이터셋 듀얼 세이브 (영구 보존본 + 연동본)
    with open(OUTPUT_VERSIONED, "w", encoding="utf-8") as f:
        for item in hc_dpo_dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    shutil.copy2(OUTPUT_VERSIONED, OUTPUT_LATEST)
            
    print(f"\n🎉 HC-DA-DPO 가중치가 포함된 데이터셋 저장 완료! (총 {len(hc_dpo_dataset)} 건)")
    print(f"   -> 보존용 영구 기록: {OUTPUT_VERSIONED}")
    print(f"   -> 다음 스크립트 연동용: {OUTPUT_LATEST}")

if __name__ == "__main__":
    main()