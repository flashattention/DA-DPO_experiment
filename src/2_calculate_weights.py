import os
import json
import torch
import shutil
import numpy as np
from tqdm import tqdm
from datetime import datetime
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

# ==========================================
# [설정] 수석 엔지니어 권장 하이퍼파라미터
# ==========================================
MODEL_ID = "Qwen/Qwen3-VL-32B-Thinking"
ALPHA = 1.0  # 🌟 기존 2.0에서 1.0으로 변경 (가중치 포화 방지 및 변별력 확보)
BETA = 1.5
LENGTH_PENALTY_ALPHA = 0.8

# 경로 설정은 기존 로직 유지
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
ORIGINAL_DATA_PATH = os.path.join(DATA_DIR, "merged_eval_data.jsonl")
INPUT_DPO_LATEST = os.path.join(DATA_DIR, "train_dpo_pairs_latest.jsonl")
INPUT_DPO_DEFAULT = os.path.join(DATA_DIR, "train_dpo_pairs.jsonl")
OUTPUT_LATEST = os.path.join(DATA_DIR, "train_experiment_pairs_latest.jsonl")

def stable_sigmoid(x):
    # 연산 가속을 위해 numpy 최적화 및 Clipping 적용
    x = np.clip(x, -50, 50)
    return 1 / (1 + np.exp(-x))

def resolve_image_path(image_path: str) -> str:
    if os.path.isabs(image_path): return image_path
    for p in [os.path.join(PROJECT_ROOT, image_path), os.path.join(DATA_DIR, image_path)]:
        if os.path.exists(p): return p
    return os.path.join(DATA_DIR, image_path)

@torch.inference_mode() # no_grad보다 강력한 최적화
def calculate_response_logprob(model, processor, prompt_text, response_text, full_image_path):
    raw_image = Image.open(full_image_path).convert("RGB")
    file_uri = f"file://{full_image_path}"
    
    # Qwen3-VL 공식 템플릿 적용
    messages_full = [
        {"role": "user", "content": [{"type": "image", "image": file_uri}, {"type": "text", "text": prompt_text}]},
        {"role": "assistant", "content": [{"type": "text", "text": response_text}]}
    ]
    messages_prompt = [
        {"role": "user", "content": [{"type": "image", "image": file_uri}, {"type": "text", "text": prompt_text}]}
    ]

    inputs_full = processor(
        text=[processor.apply_chat_template(messages_full, tokenize=False)],
        images=[raw_image], 
        return_tensors="pt"
    ).to(model.device)
    
    inputs_prompt = processor(
        text=[processor.apply_chat_template(messages_prompt, tokenize=False, add_generation_prompt=True)],
        images=[raw_image],
        return_tensors="pt"
    ).to(model.device)

    prompt_len = inputs_prompt["input_ids"].shape[1]
    
    # 🌟 Forward Pass
    outputs = model(**inputs_full)
    logits = outputs.logits 
    
    input_ids = inputs_full["input_ids"][0]
    shift_logits = logits[0, prompt_len-1:-1, :].contiguous()
    shift_labels = input_ids[prompt_len:].contiguous()
    
    log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
    token_log_probs = torch.gather(log_probs, 1, shift_labels.unsqueeze(1)).squeeze(-1)
    
    # PAD 마스킹 및 유효 토큰 추출
    pad_id = processor.tokenizer.pad_token_id if processor.tokenizer.pad_token_id is not None else -100
    valid_mask = (shift_labels != pad_id)
    valid_token_log_probs = token_log_probs[valid_mask]
    seq_length = valid_mask.sum().item()
    
    if seq_length == 0: return 0.0
    
    # 🚨 Thinking 모델용 수식 보정
    sum_logprob = valid_token_log_probs.sum().item()
    final_logprob = sum_logprob / (seq_length ** LENGTH_PENALTY_ALPHA)
    
    return final_logprob

def main():
    print(f"🚀 [수석 엔지니어 모드] Qwen3-32B Reference Model 로딩 중...")
    
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    # 🌟 VRAM 생명줄: 비전 토큰 개수 제한
    processor.image_processor.max_pixels = 512 * 512 

    # 🌟 8대 GPU 분산 로딩 (Pipeline Parallelism)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, 
        device_map="auto", # 8대의 GPU에 나눠서 로드 (OOM 원천 차단)
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    model.eval()
    
    # 데이터 로드 (선배님 기존 로직 100% 유지)
    with open(ORIGINAL_DATA_PATH, 'r', encoding='utf-8') as f:
        match_rate_dict = {json.loads(line)['metadata']['image_path']: json.loads(line).get('eval_similarity', {}).get('overall_match_rate', 1.0) for line in f}
    
    input_dpo_path = INPUT_DPO_LATEST if os.path.exists(INPUT_DPO_LATEST) else INPUT_DPO_DEFAULT
    with open(input_dpo_path, 'r', encoding='utf-8') as f:
        dpo_pairs = [json.loads(line) for line in f]
            
    experiment_dataset = []
    print(f"🧠 {len(dpo_pairs)}건의 데이터 가중치 계산 시작...")

    for item in tqdm(dpo_pairs):
        full_image_path = resolve_image_path(item["image_path"])
        if not os.path.exists(full_image_path): continue
            
        try:
            # 🌟 Logprob 계산
            lp_c = calculate_response_logprob(model, processor, item["prompt"], item["chosen"], full_image_path)
            lp_r = calculate_response_logprob(model, processor, item["prompt"], item["rejected"], full_image_path)
            
            margin = lp_c - lp_r
            mr = match_rate_dict.get(item["image_path"], 1.0) 
            hp = 1.0 - mr
            ds = stable_sigmoid(-margin) 
            
            # 🌟 7대 Ablation 전략 가중치 산출 (그대로 유지)
            item.update({
                "logprob_margin": float(margin),
                "human_penalty": float(hp),
                "match_rate": float(mr),
                "weight_dpo": 1.0,
                "weight_da_dpo": float(stable_sigmoid(-ALPHA * margin)),
                "weight_hc_dpo": float(stable_sigmoid(BETA * (mr - 0.5))),
                "weight_hc_da_dpo": float(stable_sigmoid((-ALPHA * margin) - (BETA * hp))),
                "weight_hc_da_dpo_opt1": float(stable_sigmoid((-ALPHA * margin) - (BETA * hp * ds))),
                "weight_hc_da_dpo_opt3": float(stable_sigmoid((-ALPHA * margin) + (BETA * mr))),
                "weight_hc_da_dpo_opt1_3": float(stable_sigmoid((-ALPHA * margin) - (BETA * hp * ds) + (BETA * mr)))
            })
            
            experiment_dataset.append(item)
            
            # 🌟 VRAM 단편화 제거 (매 10개 샘플마다)
            if len(experiment_dataset) % 10 == 0:
                torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"\n⚠️ Error: {e}")
            torch.cuda.empty_cache()
            continue

    # 결과 저장 (선배님 기존 방식 유지)
    with open(OUTPUT_LATEST, "w", encoding="utf-8") as f:
        for item in experiment_dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    print(f"\n🎉 가중치 데이터셋 완성! 총 {len(experiment_dataset)} 건 저장됨.")

if __name__ == "__main__":
    main()