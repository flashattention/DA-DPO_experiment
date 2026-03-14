import os
import json
import torch
import gc
import argparse
import re
import warnings
import pandas as pd
from tqdm import tqdm
from PIL import Image
from difflib import SequenceMatcher
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel

# ==========================================
# [경로 및 상수 설정]
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

TEST_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "test_data_for_eval_latest.jsonl")
MODEL_ID = "Qwen/Qwen3-VL-32B-Thinking"

# 🛡️ 방어막 1: 어떠한 텍스트도 이 길이를 넘을 수 없음
MAX_STRING_LENGTH = 8000  # Thinking 모델을 위해 8000으로 여유 확보

# ==========================================
# [유틸리티 & 파싱 함수]
# ==========================================
def get_val(val):
    return None if pd.isna(val) else val

def difficulty_to_int(diff_str):
    mapping = {"easy": 1, "medium": 2, "hard": 3}
    return mapping.get(str(diff_str).lower(), 2)

def safe_int(val, default=0):
    """결측치나 문자가 들어와도 파이프라인이 터지지 않도록 방어"""
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return default

def safe_truncate(text):
    if not isinstance(text, str):
        text = str(text)
    if len(text) > MAX_STRING_LENGTH:
        return text[:MAX_STRING_LENGTH] + "\n...[안전장치: 절사됨]"
    return text

def extract_thoughts_and_json(text):
    """Thinking 모델의 출력에서 <think> 내용과 JSON을 안전하게 분리합니다."""
    text = safe_truncate(text)
    
    # 1. 추론 과정(Thoughts) 추출
    thoughts = ""
    thought_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL | re.IGNORECASE)
    if thought_match:
        thoughts = thought_match.group(1).strip()
    
    # 2. JSON 파싱을 위해 텍스트 클리닝
    text_clean = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE).strip()
    
    if text_clean.startswith("```json"): 
        text_clean = text_clean[7:]
    elif text_clean.startswith("```"): 
        text_clean = text_clean[3:]
    if text_clean.endswith("```"): 
        text_clean = text_clean[:-3]
        
    text_clean = text_clean.strip()

    # JSON 괄호 추출 방어 로직
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

    candidates = [text_clean]
    regex_match = re.search(r'\{.*\}', text_clean, re.DOTALL)
    if regex_match: 
        candidates.append(regex_match.group(0).strip())

    balanced = _extract_balanced_object(text_clean)
    if balanced: 
        candidates.append(balanced.strip())

    parsed_json = {"error": "JSON Parsing Failed", "raw_fallback": text_clean[:200]}
    
    for candidate in candidates:
        if not candidate: continue
        try: 
            parsed_json = json.loads(candidate, strict=False)
            if "error" in parsed_json: del parsed_json["error"] # 기존 에러 키 제거
            break
        except json.JSONDecodeError:
            repaired = re.sub(r',\s*([}\]])', r'\1', candidate) # 후행 쉼표 제거 수리
            try: 
                parsed_json = json.loads(repaired, strict=False)
                if "error" in parsed_json: del parsed_json["error"]
                break
            except Exception: 
                continue

    return thoughts, parsed_json

def calculate_similarity(human, vlm):
    try:
        sim = {}
        # safe_int를 적용하여 휴먼 데이터가 비어있어도 파이프라인 생존
        sim['e_cultural_specificity_diff'] = safe_int(vlm.get('e_cultural_specificity', 0)) - safe_int(human.get('e_cultural_specificity', 0))
        sim['a1_keyword_cultural_diff'] = safe_int(vlm.get('a1_keyword_cultural', 0)) - safe_int(human.get('a1_keyword_cultural', 0))
        sim['a2_image_keyword_alignment_diff'] = safe_int(vlm.get('a2_image_keyword_alignment', 0)) - safe_int(human.get('a2_image_keyword_alignment', 0))
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
    except Exception as e:
        return {"error": f"Similarity calc failed: {str(e)}"}

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

    thought_instruction = """
[중요: 추론 과정 기록]
평가를 시작하기 전에, 반드시 <think> 태그를 사용하여 당신의 상세한 추론 과정을 먼저 단계별로 기록하세요:
1. 이미지 관찰: 이미지에서 보이는 주요 요소들
2. 키워드 분석: 키워드가 이미지와 어떻게 연결되는지
3. 질문-답변 검증: 질문이 이미지와 관련이 있는지, 답변이 정확한지
4. 문화적 특수성 판단: 1~5점 기준을 적용하여 논리적 근거 제시
5. 난이도 평가
6. 최종 판단

추론이 모두 끝난 후, 반드시 아래 평가 기준에 맞춘 JSON 데이터만을 출력하세요."""
    
    system_prompt = f"""당신은 한국 문화가 포함된 VQA(시각적 질의응답) 데이터셋의 품질을 평가하는 테스트 모드 검수자입니다.
주어진 이미지와 메타데이터를 분석하여 기본 9가지 항목을 평가하세요.{thought_instruction}

[JSON 평가 기준 키값]
- a1_keyword_cultural: 키워드가 한국 문화를 얼마나 잘 대표하는가? (1~5 정수)
- a2_image_keyword_alignment: 이미지가 키워드와 관련이 있는가? (1~5 정수)
- a3_keyword_english: 한국어 키워드를 영미권에서 지칭하는 가장 적절한 영어 단어/구문 (문자열)
- c_question_image_relevance: 질문이 이미지와 관련이 있는지 여부 ("yes" 또는 "no")
- d_answer_accuracy: 제공된 정답이 이미지와 질문에 비추어 볼 때 정확한지 여부 ("correct", "incorrect" 또는 "ambiguous")
- d_answer_accuracy_text: d_answer_accuracy가 "no" 또는 "ambiguous"라면 이유 설명 (정확하다면 null)
- e_cultural_specificity: 정답을 맞추기 위해 한국 문화/역사/제도에 대한 사전지식이 얼마나 필수적인가? (1~5 정수)
- e_human_difficulty: 이 질문에 답하기 얼마나 어려운가? ("easy", "medium", "hard")
- g_text_required: 이미지에 있는 텍스트를 읽어야만 문제를 풀 수 있는가? ("yes" 또는 "no")

⚠️ 중요: e_cultural_specificity 점수 정의 (한국 문화 사전지식의 필수 수준)

[1점] 이미지와 질문만으로 올바른 답변을 도출할 수 있는 경우
  → 외국인이 봐도 이미지와 질문 분석만으로 답할 수 있음. (예: 건축물의 재료 묻기)
[2점] 한국 문화 사전지식이 있으면 도움이 되지만, 없어도 어느정도 풀 수 있는 경우
  → 없어도 이미지와 상식만으로 부분적 답변 가능. (예: 빙수 먹는 계절 유추)
[3점] 한국 문화 사전지식이 큰 도움이 되고, 없으면 추론 난이도가 높은 경우
  → 이미지 + 한국 문화 지식이 적절히 조합되어 정답을 유추해야 함. (예: 태극기 색깔 의미)
[4점] 한국 문화에 대한 사전지식이 거의 필수인 경우
  → 유추하는 경우도 있겠지만, 한국 문화 지식이 거의 필수. (예: 장승을 세우는 이유)
[5점] 한국 문화에 대한 사전지식 없이는 절대 풀 수 없는 경우
  → 한국 고유의 역사, 제도, 관습에 대한 사전 학습 필수. (예: 역사적 사건 명칭, 문화유산 제작시기)

⚠️ 주의: 한국 문화 요소가 있다고 해서 높은 점수를 주지 마세요. 시각 정보로 답 가능하면 1점입니다!
⚠️ 주의: 출력하는 최종 JSON 데이터 내부의 문자열 값(Value)에는 절대 쌍따옴표(")나 줄바꿈을 사용하지 마세요."""

    user_prompt = f"""[데이터]
- 키워드(Keyword): {metadata.get('keyword', '')}
- 질문(Question_ko): {metadata.get('question_ko', '')}
- 질문(Question_en): {metadata.get('question_en', '')}{options_text}
- 정답(Answer): {metadata.get('answer', '')}{rationale_text}

[지시사항] 먼저 <think> 태그로 추론을 작성하고, 그 아래에 반드시 순수한 JSON 객체 하나만 출력하세요."""
    
    return system_prompt, user_prompt

def resolve_image_path(image_path: str):
    if not image_path: return None
    if os.path.isabs(image_path):
        return image_path if os.path.exists(image_path) else None

    rel_path = image_path.lstrip("./")
    candidates = [
        os.path.join(PROJECT_ROOT, rel_path),
        os.path.join(PROJECT_ROOT, "data", rel_path),
    ]
    for p in candidates:
        if os.path.exists(p): return p
    return None

def load_image_for_model(image_path: str):
    """PIL 팔레트+투명도 경고를 피하면서 모델 입력용 RGB 이미지를 반환"""
    with Image.open(image_path) as img:
        # PNG의 transparency(bytes) 정보가 있는 팔레트/그레이스케일 이미지는
        # 바로 RGB 변환 시 경고가 발생할 수 있으므로 RGBA를 거쳐 변환한다.
        transparency = img.info.get("transparency")
        if isinstance(transparency, bytes) and img.mode in {"P", "L", "1"}:
            img = img.convert("RGBA")

        # 일부 PIL 버전/포맷 조합에서 내부 재변환 경고가 남을 수 있어
        # 해당 경고 메시지에 한해 노이즈를 억제한다.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Palette images with Transparency expressed in bytes should be converted to RGBA images",
                category=UserWarning,
            )
            return img.convert("RGB")

# ==========================================
# [모델 분산 추론 실행기]
# ==========================================
def run_inference_and_update(model, processor, dataset, model_name, prompt_mode, device, rank, results_dict):
    print(f"\n🧠 [GPU {rank} | {model_name} | {prompt_mode}] 추론 시작...")
    missing_images = 0
    
    for item in tqdm(dataset, desc=f"GPU-{rank} {model_name}-{prompt_mode}"):
        metadata = item.get('metadata', {})
        human_eval = item.get('human_eval', {})
        image_path = metadata.get('image_path', '')
        
        if image_path not in results_dict: continue
            
        full_image_path = resolve_image_path(image_path)
        if not full_image_path:
            missing_images += 1
            continue

        raw_image = load_image_for_model(full_image_path)
        file_uri = f"file://{full_image_path}"
        
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
            # 🛡️ Thinking 모델의 긴 사고 과정을 보장하기 위해 4096으로 확장
            max_tokens = 4096

        text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text_input], images=[raw_image], return_tensors="pt", padding=True).to(device)
        
        try:
            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_new_tokens=max_tokens, temperature=0.1)
                
            generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
            raw_response_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            
            res_row = results_dict[image_path]
            
            if prompt_mode == "Simple":
                res_row[f"{model_name}_Simple_Response"] = safe_truncate(raw_response_text)
            else:
                # 🛡️ 텍스트에서 Thoughts와 JSON을 완벽하게 분리
                thoughts, parsed_json = extract_thoughts_and_json(raw_response_text)
                eval_sim = calculate_similarity(human_eval, parsed_json) if "error" not in parsed_json else {}
                
                # 연구자의 Ablation 분석을 위해 Thoughts를 독립된 열로 저장!
                res_row[f"{model_name}_Detailed_Thoughts"] = thoughts
                res_row[f"{model_name}_Detailed_JSON"] = safe_truncate(json.dumps(parsed_json, ensure_ascii=False))
                res_row[f"{model_name}_Detailed_e_cultural_diff"] = eval_sim.get("e_cultural_specificity_diff", "N/A")
                res_row[f"{model_name}_Detailed_overall_match"] = eval_sim.get("overall_match_rate", "N/A")

        except Exception as e:
            res_row = results_dict[image_path]
            error_msg = f"ERROR: {str(e)}"
            if prompt_mode == "Simple":
                res_row[f"{model_name}_Simple_Response"] = error_msg
            else:
                res_row[f"{model_name}_Detailed_Thoughts"] = error_msg
                res_row[f"{model_name}_Detailed_JSON"] = "{}"
                res_row[f"{model_name}_Detailed_e_cultural_diff"] = "N/A"
                res_row[f"{model_name}_Detailed_overall_match"] = "N/A"

    if missing_images > 0:
        print(f"\n🚨 [GPU {rank}] 총 {missing_images}개의 이미지를 찾지 못해 스킵되었습니다.")

def main():
    parser = argparse.ArgumentParser(description="Evaluate Ablation Models (8-way Multi-GPU)")
    parser.add_argument("--exp_dir", type=str, required=True, help="실험 폴더 경로")
    parser.add_argument("--rank", type=int, default=0, help="현재 GPU 번호")
    parser.add_argument("--world_size", type=int, default=1, help="총 GPU 대수")
    args = parser.parse_args()
    
    # 🛡️ Multi-GPU 분산 처리 시 각 프로세스가 자신의 GPU만 바라보도록 안전장치 마련
    device_id = args.rank if torch.cuda.device_count() > args.rank else 0
    device = f"cuda:{device_id}"
    
    EXP_DIR = os.path.join(PROJECT_ROOT, args.exp_dir)
    EVAL_OUTPUT_DIR = os.path.join(EXP_DIR, "eval")
    os.makedirs(EVAL_OUTPUT_DIR, exist_ok=True)
    
    TEMP_CSV_PATH = os.path.join(EVAL_OUTPUT_DIR, f"temp_res_gpu_{args.rank}.csv")
    print(f"\n🚀 [GPU {args.rank}] HC-DA-DPO 평가 파이프라인 가동 | 대상: {EXP_DIR} | 디바이스: {device}")

    full_dataset = []
    with open(TEST_DATA_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            full_dataset.append(json.loads(line))
            
    chunk_size = len(full_dataset) // args.world_size
    start_idx = args.rank * chunk_size
    end_idx = start_idx + chunk_size if args.rank != args.world_size - 1 else len(full_dataset)
    test_dataset = full_dataset[start_idx:end_idx]
    
    results_dict = {}
    for item in test_dataset:
        img_id = item['metadata'].get('image_path', '')
        if not img_id: continue
        
        results_dict[img_id] = {
            "image_id": img_id, 
            "keyword": item['metadata'].get('keyword', ''),
            "ground_truth_answer": item['metadata'].get('answer', ''),
            "ground_truth_e_cultural": item.get('human_eval', {}).get('e_cultural_specificity', '')
        }

    MODELS_TO_EVALUATE = {
        "Baseline": None,
        "Standard_DPO": [os.path.join(EXP_DIR, "output_dpo_model"), os.path.join(PROJECT_ROOT, "output_dpo_model")],
        "DA_DPO": [os.path.join(EXP_DIR, "output_da_dpo_model"), os.path.join(PROJECT_ROOT, "output_da_dpo_model")],
        "HC_DA_DPO": [os.path.join(EXP_DIR, "output_hc_da_dpo_model"), os.path.join(PROJECT_ROOT, "output_hc_da_dpo_model")]
    }

    PROMPT_MODES = ["Simple", "Detailed"]

    print(f"\n[GPU {args.rank}] ⏳ 프로세서와 베이스 모델 로드...")
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    base_model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, device_map=device, dtype=torch.bfloat16, trust_remote_code=True
    )

    for model_name, adapter_paths in MODELS_TO_EVALUATE.items():
        print(f"\n=============================================")

        adapter_path = None
        if isinstance(adapter_paths, list):
            adapter_path = next((p for p in adapter_paths if os.path.exists(p)), None)
            if adapter_path is None and model_name != "Baseline":
                print(f"⚠️ [GPU {args.rank}] 어댑터 폴더가 없습니다. 스킵: {model_name}")
                continue
        elif adapter_paths and not os.path.exists(adapter_paths):
            print(f"⚠️ [GPU {args.rank}] 어댑터 폴더가 없습니다. 스킵: {model_name}")
            continue
        else:
            adapter_path = adapter_paths

        if model_name == "Baseline":
            print(f"⏳ [GPU {args.rank}] Baseline 로드 완료")
            model = base_model
        else:
            print(f"🔄 [GPU {args.rank}] 어댑터 장착 중: {model_name}")
            model = PeftModel.from_pretrained(base_model, adapter_path)
            
        model.eval()
        
        for prompt_mode in PROMPT_MODES:
            run_inference_and_update(model, processor, test_dataset, model_name, prompt_mode, device, args.rank, results_dict)
            
            final_df = pd.DataFrame(list(results_dict.values()))
            final_df.to_csv(TEMP_CSV_PATH, index=False, encoding='utf-8-sig')
        
        if model_name != "Baseline":
            print(f"🗑️ [GPU {args.rank}] 어댑터 해제 및 VRAM 정리: {model_name}")
            # PEFT 모델 초기화 및 OS 레벨 메모리 반환
            model.unload() 
            del model
            
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\n✅ GPU {args.rank} 작업 완료. 임시 결과 저장: {TEMP_CSV_PATH}")

if __name__ == "__main__":
    main()