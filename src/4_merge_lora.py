import os
import json
import torch
import gc
import argparse
import pandas as pd
from tqdm import tqdm
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
from peft import PeftModel

# ==========================================
# [경로 설정]
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# 🌟 1번 스크립트에서 격리해둔 최신 Test 데이터 (JSONL 포맷)
TEST_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "test_data_for_eval_latest.jsonl")
MODEL_ID = "Qwen/Qwen3-VL-32B-Thinking"

def generate_responses(model, processor, dataset, model_name):
    """특정 모델을 사용하여 Test Set의 답변을 생성합니다."""
    results = []
    print(f"\n🧠 [{model_name}] 추론 시작...")
    
    for item in tqdm(dataset, desc=f"Evaluating {model_name}"):
        metadata = item.get('metadata', {})
        image_path = metadata.get('image_path')
        full_image_path = os.path.join(PROJECT_ROOT, image_path)
        
        # 안전장치: 이미지 파일이 없으면 스킵
        if not os.path.exists(full_image_path):
            print(f"⚠️ 경고: 이미지 없음 -> {full_image_path}")
            continue
            
        # 🌟 [버그 픽스 1] PIL Image 객체를 사용하여 Base64 파싱 에러(Incorrect padding) 원천 차단
        raw_image = Image.open(full_image_path).convert("RGB")
        file_uri = f"file://{full_image_path}"
        
        # 🌟 [버그 픽스 2] DPO 변환 전 원본 메타데이터를 사용해 평가용 프롬프트 실시간 조립
        question = metadata.get('question_ko', '')
        options = metadata.get('options', '없음')
        if isinstance(options, list) and len(options) == 0: options = "없음 (주관식)"
        
        prompt_text = f"다음 이미지와 질문을 보고 적절한 답을 하시오.\n[질문]: {question}\n[보기]: {options}"
        
        messages = [
            {"role": "user", "content": [{"type": "image", "image": file_uri}, {"type": "text", "text": prompt_text}]}
        ]
        text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # raw_image 전달
        inputs = processor(text=[text_input], images=[raw_image], return_tensors="pt", padding=True).to(model.device)
        
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=512, temperature=0.1)
            
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        
        results.append({
            "image_id": image_path,
            f"{model_name}_response": response
        })
    return pd.DataFrame(results)

def main():
    # 🌟 [실험 격리 적용] 터미널에서 평가할 실험 폴더 경로를 받아옴
    parser = argparse.ArgumentParser(description="Evaluate Ablation Models")
    parser.add_argument("--exp_dir", type=str, required=True, help="평가할 모델이 있는 실험 폴더 경로 (예: runs/exp_2026...)")
    args = parser.parse_args()
    
    EXP_DIR = os.path.join(PROJECT_ROOT, args.exp_dir)
    EVAL_OUTPUT_DIR = os.path.join(EXP_DIR, "eval")
    os.makedirs(EVAL_OUTPUT_DIR, exist_ok=True)
    
    OUTPUT_CSV_PATH = os.path.join(EVAL_OUTPUT_DIR, "evaluation_results.csv")
    print(f"\n🚀 [평가 시작] 대상 실험 폴더: {EXP_DIR}")

    # 🌟 [버그 픽스 3] JSONL 파일 한 줄씩 제대로 읽기
    test_dataset = []
    with open(TEST_DATA_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            test_dataset.append(json.loads(line))
            
    # 채점 기준이 될 정답(Ground Truth) 데이터 구성
    final_df = pd.DataFrame([{
        "image_id": item['metadata']['image_path'], 
        "ground_truth_answer": item['metadata']['answer'],
        "ground_truth_rationale": item['metadata'].get('rationale_ko', '')
    } for item in test_dataset])

    # 평가할 모델 리스트 (받아온 실험 폴더 기준 동적 맵핑)
    MODELS_TO_EVALUATE = {
        "Baseline": None,
        "Standard_DPO": os.path.join(EXP_DIR, "output_dpo_model"),
        "DA_DPO": os.path.join(EXP_DIR, "output_da_dpo_model"),
        "HC_DA_DPO": os.path.join(EXP_DIR, "output_hc_da_dpo_model")
    }

    # 4개의 모델을 순차적으로 로드하며 추론
    for model_name, adapter_path in MODELS_TO_EVALUATE.items():
        print(f"\n=============================================")
        print(f"⏳ 로딩 중... 모델: {model_name}")
        
        if adapter_path and not os.path.exists(adapter_path):
            print(f"⚠️ [스킵] 어댑터 폴더가 없습니다. 학습 누락 의심: {adapter_path}")
            continue

        processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True
        )
        
        if adapter_path:
            print(f"🔗 어댑터 결합 중: {adapter_path}")
            model = PeftModel.from_pretrained(base_model, adapter_path)
        else:
            print(f"🧠 Base 모델 단독 사용")
            model = base_model
            
        model.eval()
        
        df_results = generate_responses(model, processor, test_dataset, model_name)
        
        final_df = pd.merge(final_df, df_results[["image_id", f"{model_name}_response"]], on="image_id", how="left")
        
        # 안전한 중간 저장 (실험 폴더 내부에 종속 저장됨)
        final_df.to_csv(OUTPUT_CSV_PATH, index=False, encoding='utf-8-sig')
        
        # VRAM 완벽 초기화
        del model
        if adapter_path: del base_model
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\n🎉 모든 모델 추론 완료! 결과가 다음 경로에 안전하게 저장되었습니다.")
    print(f"📁 결과물 경로: {OUTPUT_CSV_PATH}")

if __name__ == "__main__":
    main()