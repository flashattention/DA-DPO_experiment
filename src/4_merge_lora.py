import os
import json
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor
from peft import PeftModel

# ==========================================
# [경로 설정]
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODEL_DIR = os.path.join(PROJECT_ROOT, "output_da_dpo_model", "final_adapter")

TEST_DATA_PATH = os.path.join(DATA_DIR, "test_data_for_eval.jsonl")
RESULT_OUTPUT_PATH = os.path.join(DATA_DIR, "eval_results_comparison.csv")

# 베이스 모델 (3번 스크립트와 동일)
BASE_MODEL_ID = "Qwen/Qwen3-VL-32B-Thinking"

def main():
    print("🚀 평가 파이프라인 시작: Base Model + LoRA Adapter 로딩 중...")
    
    # 1. Base Model 로드 (추론용이므로 bfloat16으로 빠르게 로드)
    processor = AutoProcessor.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    
    # 2. LoRA Adapter 병합 (PeftModel)
    print(f"🔗 LoRA 어댑터 병합 중... (경로: {MODEL_DIR})")
    model = PeftModel.from_pretrained(base_model, MODEL_DIR)
    model.eval()
    print("✅ 모델 준비 완료!")

    # 3. Test 데이터 로드
    test_data = []
    with open(TEST_DATA_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            test_data.append(json.loads(line))
            
    print(f"📊 총 {len(test_data)}개의 Test 데이터 평가를 시작합니다.")

    results = []

    # 4. 추론 및 채점 루프
    for item in tqdm(test_data, desc="Evaluating Test Set"):
        metadata = item['metadata']
        human_eval = item['human_eval']
        baseline_eval = item['vlm_eval'] # 파인튜닝 전 기존 모델의 평가 결과
        
        image_path = os.path.join(PROJECT_ROOT, metadata['image_path'])
        
        # 프롬프트 구성 (1번 스크립트의 포맷과 동일하게)
        prompt_text = f"""다음 이미지와 한국 문화 VQA 데이터를 보고 적절성을 평가하시오.
[키워드]: {metadata['keyword']}
[질문]: {metadata['question_ko']}
[보기]: {metadata.get('options', '없음')}
[정답]: {metadata['answer']}"""

        # Qwen-VL ChatML 포맷 적용
        messages = [
            {"role": "user", "content": [{"type": "image", "image": image_path}, {"type": "text", "text": prompt_text}]}
        ]
        
        text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text_prompt], images=[image_path], return_tensors="pt", padding=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # 모델 추론 (Generation)
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs, 
                max_new_tokens=512, 
                temperature=0.1, # 일관된 평가를 위해 온도를 낮춤
                do_sample=False
            )
            
        # 입력 프롬프트 부분 잘라내고 순수 생성 결과만 추출
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]
        output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]
        
        # 5. 결과 기록
        # (실제 논문에서는 여기서 JSON 파싱을 통해 각 항목별 점수 오차(MSE 등)를 세밀하게 구합니다)
        results.append({
            "image_id": metadata['image_path'],
            "difficulty": human_eval.get('e_human_difficulty', 'N/A'),
            "human_score_accuracy": human_eval.get('d_answer_accuracy', 0),
            "baseline_score_accuracy": baseline_eval.get('d_answer_accuracy', 0),
            "new_model_output": output_text, # 새 모델이 생성한 JSON 및 <think> 텍스트
            "baseline_thought": baseline_eval.get('thought_trace', ''),
        })

    # 6. 결과 저장
    df_results = pd.DataFrame(results)
    df_results.to_csv(RESULT_OUTPUT_PATH, index=False, encoding='utf-8-sig')
    
    print(f"\n🎉 평가 완료! 결과가 엑셀 형태({RESULT_OUTPUT_PATH})로 저장되었습니다.")
    print("다음 단계: 이 CSV 파일을 분석하여 논문에 들어갈 그래프(Hard vs Easy 성능 향상 폭 등)를 그릴 수 있습니다.")

if __name__ == "__main__":
    main()