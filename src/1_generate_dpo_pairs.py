import os
import json
import asyncio
import pandas as pd
from sklearn.model_selection import train_test_split
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

# ==========================================
# [경로 자동 설정] 스크립트 위치 기반으로 완벽한 경로 추적
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))       # 현재 스크립트(src) 폴더 경로
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)                    # 최상위(DA-DPO_EXPERIMENT) 폴더 경로
DATA_DIR = os.path.join(PROJECT_ROOT, "data")                 # data 폴더 경로

INPUT_FILE_PATH = os.path.join(DATA_DIR, "merged_eval_data.jsonl")
TRAIN_OUTPUT_PATH = os.path.join(DATA_DIR, "train_dpo_pairs.jsonl")
TEST_OUTPUT_PATH = os.path.join(DATA_DIR, "test_data_for_eval.jsonl")

# ==========================================
# [설정] 로컬 vLLM 서버 세팅 (A100 8대 환경)
# ==========================================
client = AsyncOpenAI(
    api_key="EMPTY", 
    base_url="http://localhost:8000/v1"
)

# vLLM 서버 구동 시 사용한 모델 이름
GHOSTWRITER_MODEL = "Qwen/Qwen3-VL-235B-A22B-Thinking" 

MAX_CONCURRENT_REQUESTS = 50 
semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

# ==========================================
# 유틸리티 함수: 보기(Options) 빈 배열 처리
# ==========================================
def format_options(options):
    if options in ["[]", "['']", "[\"\"]", "", None] or (isinstance(options, list) and len(options) == 0):
        return "없음 (주관식/단답형)"
    return options

# ==========================================
# 1. 데이터 불러오기 및 Train/Test 분리
# ==========================================
print(f"데이터 로딩 중... 경로: {INPUT_FILE_PATH}")
with open(INPUT_FILE_PATH, 'r', encoding='utf-8') as f:
    raw_data = [json.loads(line) for line in f]

df = pd.DataFrame(raw_data)

# 데이터 누수 방지를 위해 '이미지' 기준으로 90:10 분리
df['image_path'] = df['metadata'].apply(lambda x: x['image_path'])
unique_images = df['image_path'].unique().tolist()  # pyarrow 호환성을 위해 리스트로 변환
train_images, test_images = train_test_split(unique_images, test_size=0.1, random_state=42)

train_df = df[df['image_path'].isin(train_images)].copy()
test_df = df[df['image_path'].isin(test_images)].copy()

# Test 데이터 보관
test_df.to_json(TEST_OUTPUT_PATH, orient="records", lines=True, force_ascii=False)
print(f"✅ Test 데이터 격리 완료 및 저장됨 (총 {len(test_df)}건) -> {TEST_OUTPUT_PATH}")

# ==========================================
# 2. Qwen3-235B를 이용한 Chosen <think> 대필 (비동기)
# ==========================================
async def generate_chosen_think(metadata, human_eval):
    async with semaphore:
        system_prompt = """당신은 한국 문화 전문가이자 AI 모델의 평가 기준을 설계하는 최고 권위자입니다.
주어진 VQA 문제에 대해 '인간 평가자'가 왜 이런 점수(JSON)를 주었는지 그 논리적인 사고 과정(Thought Trace)을 역추적하여 작성해야 합니다.
반드시 <think> 로 시작해서 </think> 로 끝나는 3~4문장의 추론 과정을 한국어로 작성하세요."""

        options_text = format_options(metadata.get('options'))

        user_prompt = f"""
[VQA 문제 정보]
- 키워드: {metadata['keyword']}
- 질문: {metadata['question_ko']}
- 보기: {options_text}
- 정답: {metadata['answer']}
- 해설: {metadata.get('rationale_ko', '없음')}

[인간 평가자의 실제 채점 결과]
- 문화 특수성 (e_cultural_specificity): {human_eval.get('e_cultural_specificity', 'N/A')}점
- 인간 체감 난이도 (e_human_difficulty): {human_eval.get('e_human_difficulty', 'N/A')}
- 질문/이미지 관련성: {human_eval.get('c_question_image_relevance', 'N/A')}
- 정답 정확성: {human_eval.get('d_answer_accuracy', 'N/A')}

위 평가 결과가 도출되기까지의 올바른 문화적, 시각적, 논리적 사고 과정을 <think> 태그 안에 작성해주세요."""

        try:
            response = await client.chat.completions.create(
                model=GHOSTWRITER_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=300
            )
            think_trace = response.choices[0].message.content.strip()
            
            if not think_trace.startswith("<think>"):
                think_trace = f"<think>\n{think_trace}\n</think>"
            return think_trace
            
        except Exception as e:
            print(f"API Error: {e}")
            return "<think>\n주어진 정보와 문화적 맥락을 종합하여 위와 같이 평가함.\n</think>"

async def process_dpo_pairs(dataframe):
    tasks = []
    for _, row in dataframe.iterrows():
        metadata = row['metadata']
        human_eval = row['human_eval']
        tasks.append(generate_chosen_think(metadata, human_eval))
    
    chosen_thinks = await tqdm_asyncio.gather(*tasks, desc=f"Generating <think> traces with {GHOSTWRITER_MODEL}")
    
    dpo_dataset = []
    for i, (_, row) in enumerate(dataframe.iterrows()):
        metadata = row['metadata']
        human_eval = row['human_eval']
        vlm_eval = row['vlm_eval']
        eval_sim = row.get('eval_similarity', {})
        
        # 100% 일치하는 쉬운 케이스 제외
        if eval_sim.get('overall_match_rate', 0) == 1.0:
            continue
            
        options_text = format_options(metadata.get('options'))
        
        prompt_input = f"""다음 이미지와 한국 문화 VQA 데이터를 보고 적절성을 평가하시오.
[키워드]: {metadata['keyword']}
[질문]: {metadata['question_ko']}
[보기]: {options_text}
[정답]: {metadata['answer']}"""

        chosen_json_str = json.dumps(human_eval, ensure_ascii=False)
        chosen_text = f"{chosen_thinks[i]}\n{chosen_json_str}"
        
        baseline_think = vlm_eval.pop('thought_trace', '추론 과정 없음')
        rejected_think = f"<think>\n{baseline_think}\n</think>"
        rejected_json_str = json.dumps(vlm_eval, ensure_ascii=False)
        rejected_text = f"{rejected_think}\n{rejected_json_str}"
        
        dpo_dataset.append({
            "image_path": metadata['image_path'],
            "prompt": prompt_input,
            "chosen": chosen_text,
            "rejected": rejected_text
        })
        
    return dpo_dataset

# ==========================================
# 3. 메인 실행 블록
# ==========================================
if __name__ == "__main__":
    print(f"🔥 로컬 vLLM 서버에 연결하여 DPO 데이터 생성을 시작합니다...")
    loop = asyncio.get_event_loop()
    final_dpo_data = loop.run_until_complete(process_dpo_pairs(train_df))
    
    with open(TRAIN_OUTPUT_PATH, "w", encoding="utf-8") as f:
        for item in final_dpo_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    print(f"🎉 완벽한 DPO 데이터셋 저장 완료! ({TRAIN_OUTPUT_PATH}, 총 {len(final_dpo_data)} Pairs 생성됨)")