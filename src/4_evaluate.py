import os
import json
import torch
import gc
import argparse
import re
import pandas as pd
from tqdm import tqdm
from PIL import Image
from difflib import SequenceMatcher
from transformers import AutoProcessor, AutoModelForCausalLM
from peft import PeftModel

# ==========================================
# [경로 설정]
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# 🌟 1번 스크립트에서 격리해둔 최신 Test 데이터
TEST_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "test_data_for_eval_latest.jsonl")
MODEL_ID = "Qwen/Qwen3-VL-32B-Thinking"

# ==========================================
# [유틸리티 & 파싱 함수 (원본 100% 복원)]
# ==========================================
def get_val(val):
    return None if pd.isna(val) else val

def difficulty_to_int(diff_str):
    mapping = {"easy": 1, "medium": 2, "hard": 3}
    return mapping.get(str(diff_str).lower(), 2)

def safe_parse_json(text):
    """VLM이 뱉은 텍스트에서 찌꺼기를 제거하고 안전하게 JSON만 추출합니다."""
    def _extract_balanced_object(s: str):
        start = s.find('{')
        if start == -1: return None
        depth = 0; in_string = False; escape = False
        for i in range(start, len(s)):
            ch = s[i]
            if escape: escape = False; continue
            if ch == '\\': escape = True; continue
            if ch == '"': in_string = not in_string; continue
            if in_string: continue
            if ch == '{': depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0: return s[start:i + 1]
        return None

    text = re.sub(r'<think>.*?</think>', '', str(text), flags=re.DOTALL).strip()
    if text.startswith("```json"): 
        text = text[7:]
    elif text.startswith("```"): 
        text = text[3:]
    if text.endswith("```"): 
        text = text[:-3]
        
    text = text.strip()

    candidates = [text]
    regex_match = re.search(r'\{.*\}', text, re.DOTALL)
    if regex_match: 
        candidates.append(regex_match.group(0).strip())

    balanced = _extract_balanced_object(text)
    if balanced: 
        candidates.append(balanced.strip())

    uniq_candidates = []
    seen = set()
    for c in candidates:
        if c and c not in seen:
            uniq_candidates.append(c)
            seen.add(c)

    for candidate in uniq_candidates:
        try: 
            return json.loads(candidate, strict=False)
        except json.JSONDecodeError:
            repaired = re.sub(r',\s*([}\]])', r'\1', candidate)
            try: 
                return json.loads(repaired, strict=False)
            except Exception: 
                continue

    return {"error": "JSON Parsing Failed", "raw": text[:200]}

def calculate_similarity(human, vlm):
    """인간 평가와 VLM 평가 간의 차이/유사도를 계산합니다."""
    sim = {}
    sim['e_cultural_specificity_diff'] = int(vlm.get('e_cultural_specificity', 0)) - int(human.get('e_cultural_specificity', 0))
    sim['a1_keyword_cultural_diff'] = int(vlm.get('a1_keyword_cultural', 0)) - int(human.get('a1_keyword_cultural', 0))
    sim['a2_image_keyword_alignment_diff'] = int(vlm.get('a2_image_keyword_alignment', 0)) - int(human.get('a2_image_keyword_alignment', 0))
    sim['e_human_difficulty_diff'] = difficulty_to_int(vlm.get('e_human_difficulty')) - difficulty_to_int(human.get('e_human_difficulty'))
    
    sim['c_question_image_relevance_match'] = str(vlm.get('c_question_image_relevance')).lower() == str(human.get('c_question_image_relevance')).lower()
    sim['g_text_required_match'] = str(vlm.get('g_text_required')).lower() == str(human.get('g_text_required')).lower()
    
    h_acc = str(human.get('d_answer_accuracy')).lower()
    v_acc = str(vlm.get('d_answer_accuracy')).lower()
    
    if h_acc == v_acc: sim['d_answer_accuracy_score'] = 1.0
    elif "ambiguous" in [h_acc, v_acc]: sim['d_answer_accuracy_score'] = 0.5
    else: sim['d_answer_accuracy_score'] = 0.0
    
    h_a3 = str(human.get('a3_keyword_english', ''))
    v_a3 = str(vlm.get('a3_keyword_english', ''))
    sim['a3_keyword_english_similarity'] = round(SequenceMatcher(None, h_a3.lower(), v_a3.lower()).ratio(), 4)

    scores = [
        1.0 if sim['c_question_image_relevance_match'] else 0.0,
        sim['d_answer_accuracy_score'],
        1.0 if sim['g_text_required_match'] else 0.0,
        max(0, 1.0 - (abs(sim['a1_keyword_cultural_diff']) / 4.0)),
        max(0, 1.0 - (abs(sim['a2_image_keyword_alignment_diff']) / 4.0)),
        max(0, 1.0 - (abs(sim['e_cultural_specificity_diff']) / 4.0)),
        max(0, 1.0 - (abs(sim['e_human_difficulty_diff']) / 2.0)),
        sim['a3_keyword_english_similarity']
    ]
    sim['overall_match_rate'] = round(sum(scores) / 8.0, 4)
    return sim

# ==========================================
# [프롬프트 생성기]
# ==========================================
def build_simple_prompt(metadata):
    question = metadata.get('question_ko', '')
    options = metadata.get('options', '없음')
    if isinstance(options, list) and len(options) == 0: options = "없음 (주관식)"
    return f"다음 이미지와 질문을 보고 적절한 답을 하시오.\n[질문]: {question}\n[보기]: {options}"

def build_detailed_prompt(metadata):
    options_val = get_val(metadata.get('options'))
    options_text = f"\n- 보기(Options): {options_val}" if pd.notna(options_val) else ""
    rationale_val = get_val(metadata.get('rationale_ko'))
    rationale_text = f"\n- 정답 해설(Rationale): {rationale_val}" if pd.notna(rationale_val) else ""

    system_prompt = """당신은 한국 문화가 포함된 VQA(시각적 질의응답) 데이터셋의 품질을 평가하는 테스트 모드 검수자입니다.
주어진 이미지와 메타데이터(키워드, 질문, 정답 등)를 간단히 분석하여 기본 9가지 항목을 평가하세요.

[중요: 추론 과정 기록]
평가를 시작하기 전에, 반드시 <think> 태그 내에서 당신의 상세한 추론 과정을 단계별로 기록하세요:
1. 이미지 관찰: 이미지에서 보이는 주요 요소들
2. 키워드 분석: 키워드가 이미지와 어떻게 연결되는지
3. 질문-답변 검증: 질문이 이미지와 관련이 있는지, 답변이 정확한지
4. 문화적 특수성 판단: 각 e_cultural_specificity 점수 레벨(1~5점)을 고려하며, 왜 특정 점수를 선택했는지 구체적으로 설명
5. 난이도 평가: 사람이 이 문제를 풀 때의 어려움
6. 최종 판단: 각 항목에 대한 최종 점수와 그 근거

예시 형식:
<think>
[이미지 관찰] 이미지에는 ... 이 보입니다.
[키워드 분석] 키워드 '...'는 ...을 의미하며, 이미지의 ...와 일치합니다.
[문화적 특수성] 이 질문은 ...하기 때문에 e_cultural_specificity는 X점입니다. 왜냐하면 ...
</think>

[평가 기준]
- a1_keyword_cultural: 키워드가 한국 문화를 얼마나 잘 대표하는가? (1~5 정수)
- a2_image_keyword_alignment: 이미지가 키워드와 관련이 있는가? (1~5 정수)
- a3_keyword_english: 한국어 키워드의 영어 번역 (문자열)
- c_question_image_relevance: 질문 관련성 ("yes" 또는 "no")
- d_answer_accuracy: 정답 정확성 ("correct", "incorrect" 또는 "ambiguous")
- d_answer_accuracy_text: 오답일 경우 이유
- e_cultural_specificity: 정답을 맞추기 위해 한국 문화/역사/제도 사전지식이 얼마나 필수적인가? (1~5 정수)
- e_human_difficulty: 난이도 ("easy", "medium", "hard")
- g_text_required: 텍스트 읽기 필수 여부 ("yes" 또는 "no")
- thought_trace: 반드시 포함. <think>...</think> 형식 문자열

⚠️ 중요: e_cultural_specificity 점수 정의
[1점] 이미지와 질문만으로 올바른 답변 도출 가능 (외국인도 가능)
[2점] 지식이 있으면 도움되나, 없어도 어느정도 풀 수 있음
[3점] 지식이 큰 도움이 되고, 없으면 추론 난이도 높음
[4점] 사전지식이 거의 필수
[5점] 사전지식 없이는 절대 풀 수 없음

반드시 JSON 형식으로만 답변을 출력하세요. 
⚠️ 중요: e_cultural_specificity 점수 정의 (한국 문화 사전지식의 필수 수준)

[1점] 이미지와 질문만으로 올바른 답변을 도출할 수 있는 경우
  → 한국 문화와 깊게 관련되어 있는지 여부와 관계 없이 이미지와 질문의 정보만으로 답변 가능한 경우.
  → 외국인이 봐도 이미지와 질문 분석만으로 답할 수 있음.
  예: "이 사진의 전통 의상을 입은 사람이 서 있는 건축물의 재료는?" (나무, 벽돌 등 건축물 시각 정보만으로 답함)

[2점] 한국 문화 사전지식이 있으면 도움이 되지만, 없어도 어느정도 풀 수 있는 경우
  → 사전지식이 있으면 더 정확하고 쉽게 답할 수 있음.
  → 없어도 이미지와 상식만으로 부분적 답변 가능.
  예: "한국에서 팥빙수를 주로 언제 먹나요?" (사진속 재료에 얼음이 들어가 일반적 추측으로 어느 정도 답 가능)

[3점] 한국 문화 사전지식이 큰 도움이 되고, 없으면 추론 난이도가 높은 경우
  → 지식이 있으면 쉽고, 없으면 매우 어려움.
  → 이미지 + 한국 문화 지식이 적절히 조합되어야 함.
  예: "이 대한민국 국기의 색깔들이 나타내는 의미는?" (사진속 국기를 보고 해와 바다, 음과 양 등을 유추할 수도 있지만 한국 국기 색깔의 의미에 대한 사전 지식이 있어야 정확하게 답변 가능)

[4점] 한국 문화에 대한 사전지식이 거의 필수인 경우
  → 한국 문화 지식이 있으면 쉽게 답할 수 있음.
  → 지식이 없으면 답하기 매우 어려움.
  예: "사진처럼 마을 앞에 장승을 세워두는 이유는?" (유추하는 경우도 있겠지만, 한국 문화 지식이 거의 필수)

[5점] 한국 문화에 대한 사전지식 없이는 절대 풀 수 없는 경우
  → 한국을 배경지식 없이 이미지나 질문만 봐서는 답할 수 없음.
  → 한국 고유의 역사, 제도, 관습에 대한 사전 학습 필수.
  예: "한국 역사의 이 사건의 명칭은?" (한국사 지식 절대 필수)
  예: "한국 고유의 이 문화유산의 제작시기는?" (한국 문화유산 지식 절대 필수)

⚠️ 중요: 한국 문화 요소가 있다고 해서 높은 점수를 주지 마세요. 이미지와 질문만으로 답 가능하면 1점입니다!
⚠️ 중요: JSON의 문자열 값(Value) 내부에는 절대 쌍따옴표(")나 줄바꿈을 사용하지 마세요.

⚠️ 중요: --save_thoughts 옵션이 활성화되었으므로 thought_trace 키를 반드시 포함하고, 값은 <think> ... </think> 형식 문자열로 작성하세요.

반드시 JSON 형식으로만 답변을 출력하세요."""

    user_prompt = f"""[데이터]
- 키워드(Keyword): {metadata.get('keyword', '')}
- 질문(Question_ko): {metadata.get('question_ko', '')}
- 질문(Question_en): {metadata.get('question_en', '')}{options_text}
- 정답(Answer): {metadata.get('answer', '')}{rationale_text}

[지시사항] --save_thoughts 활성화 상태입니다. JSON 내부 thought_trace 값에 <think>...</think> 형식으로 추론 경로를 넣고, 나머지 평가 필드도 함께 출력하세요."""
    
    return system_prompt, user_prompt

# ==========================================
# [모델 추론 실행기]
# ==========================================
def generate_responses(model, processor, dataset, model_name, prompt_mode):
    """
    prompt_mode: "Simple" 또는 "Detailed"
    """
    results = []
    print(f"\n🧠 [{model_name} | {prompt_mode} Prompt] 추론 시작...")
    
    for item in tqdm(dataset, desc=f"{model_name}-{prompt_mode}"):
        metadata = item.get('metadata', {})
        human_eval = item.get('human_eval', {})
        image_path = metadata.get('image_path')
        full_image_path = os.path.join(PROJECT_ROOT, image_path)
        
        if not os.path.exists(full_image_path):
            continue
            
        raw_image = Image.open(full_image_path).convert("RGB")
        file_uri = f"file://{full_image_path}"
        
        # 프롬프트 모드에 따른 메시지 조립
        if prompt_mode == "Simple":
            prompt_text = build_simple_prompt(metadata)
            messages = [{"role": "user", "content": [{"type": "image", "image": file_uri}, {"type": "text", "text": prompt_text}]}]
            max_tokens = 512
        else:
            sys_prompt, user_prompt = build_detailed_prompt(metadata)
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": [{"type": "image", "image": file_uri}, {"type": "text", "text": user_prompt}]}
            ]
            max_tokens = 2048

        text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text_input], images=[raw_image], return_tensors="pt", padding=True).to(model.device)
        
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=max_tokens, temperature=0.1)
            
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        response_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        
        # 결과 저장 로직 분기
        row_result = {"image_id": image_path}
        
        if prompt_mode == "Simple":
            row_result[f"{model_name}_Simple_Response"] = response_text
        else:
            parsed_json = safe_parse_json(response_text)
            eval_sim = calculate_similarity(human_eval, parsed_json) if "error" not in parsed_json else {}
            
            row_result[f"{model_name}_Detailed_Raw"] = response_text
            row_result[f"{model_name}_Detailed_JSON"] = json.dumps(parsed_json, ensure_ascii=False)
            row_result[f"{model_name}_Detailed_e_cultural_diff"] = eval_sim.get("e_cultural_specificity_diff", "N/A")
            row_result[f"{model_name}_Detailed_overall_match"] = eval_sim.get("overall_match_rate", "N/A")
            
        results.append(row_result)
        
    return pd.DataFrame(results)

def main():
    parser = argparse.ArgumentParser(description="Evaluate Ablation Models (8-way)")
    parser.add_argument("--exp_dir", type=str, required=True, help="평가할 모델이 있는 실험 폴더 경로 (예: runs/exp_2026...)")
    args = parser.parse_args()
    
    EXP_DIR = os.path.join(PROJECT_ROOT, args.exp_dir)
    EVAL_OUTPUT_DIR = os.path.join(EXP_DIR, "eval")
    os.makedirs(EVAL_OUTPUT_DIR, exist_ok=True)
    
    OUTPUT_CSV_PATH = os.path.join(EVAL_OUTPUT_DIR, "evaluation_results_8way.csv")
    print(f"\n🚀 [8-way Ablation 평가 시작] 대상 실험 폴더: {EXP_DIR}")

    test_dataset = []
    with open(TEST_DATA_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            test_dataset.append(json.loads(line))
            
    # 정답(Ground Truth) 베이스 데이터프레임 구성
    final_df = pd.DataFrame([{
        "image_id": item['metadata']['image_path'], 
        "keyword": item['metadata'].get('keyword', ''),
        "ground_truth_answer": item['metadata'].get('answer', ''),
        "ground_truth_e_cultural": item.get('human_eval', {}).get('e_cultural_specificity', '')
    } for item in test_dataset])

    MODELS_TO_EVALUATE = {
        "Baseline": None,
        "Standard_DPO": os.path.join(EXP_DIR, "output_dpo_model"),
        "DA_DPO": os.path.join(EXP_DIR, "output_da_dpo_model"),
        "HC_DA_DPO": os.path.join(EXP_DIR, "output_hc_da_dpo_model")
    }

    PROMPT_MODES = ["Simple", "Detailed"]

    # 4개 모델 x 2개 프롬프트 = 총 8가지 경우의 수 순차 진행
    for model_name, adapter_path in MODELS_TO_EVALUATE.items():
        print(f"\n=============================================")
        print(f"⏳ 로딩 중... 모델: {model_name}")
        
        if adapter_path and not os.path.exists(adapter_path):
            print(f"⚠️ [스킵] 어댑터 폴더가 없습니다: {adapter_path}")
            continue

        processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True
        )
        
        if adapter_path:
            model = PeftModel.from_pretrained(base_model, adapter_path)
        else:
            model = base_model
            
        model.eval()
        
        # 각 모델별로 Simple과 Detailed 프롬프트를 연속해서 테스트
        for prompt_mode in PROMPT_MODES:
            df_results = generate_responses(model, processor, test_dataset, model_name, prompt_mode)
            
            # DataFrame 병합 (기존에 존재하는 image_id 기준)
            final_df = pd.merge(final_df, df_results, on="image_id", how="left")
            
            # 중간중간 안전하게 CSV 덮어쓰기 저장
            final_df.to_csv(OUTPUT_CSV_PATH, index=False, encoding='utf-8-sig')
        
        # 한 모델의 평가(2가지 모드)가 끝나면 메모리 완벽 정리
        del model
        if adapter_path: 
            del base_model
            
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\n🎉 8-way Ablation 추론 완벽 종료! 결과가 다음 경로에 저장되었습니다.")
    print(f"📁 결과물 경로: {OUTPUT_CSV_PATH}")

if __name__ == "__main__":
    main()