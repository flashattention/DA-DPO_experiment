import asyncio
import base64
import math
import os
import pandas as pd
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm
from PIL import Image
import io

# vLLM 비동기 클라이언트 설정
aclient = AsyncOpenAI(
    api_key="EMPTY", 
    base_url="http://localhost:8000/v1"
)

MODEL_NAME = "qwen3-vl-thinking" 
CONCURRENCY_LIMIT = 4 
semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)

Image.MAX_IMAGE_PIXELS = None

def encode_image_to_base64(image_path, max_size=1024):
    try:
        if not os.path.exists(image_path):
            return None
        with Image.open(image_path) as img:
            if img.mode in ('RGBA', 'LA', 'P'):
                img = img.convert('RGBA')
                bg = Image.new('RGB', img.size, (255, 255, 255))
                bg.paste(img, mask=img.split()[3])
                img = bg
            elif img.mode != 'RGB':
                img = img.convert('RGB')
            
            img.thumbnail((max_size, max_size))
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG", quality=85)
            return base64.b64encode(buffered.getvalue()).decode('utf-8')
    except Exception as e:
        print(f"이미지 변환 에러 ({image_path}): {e}")
        return None

async def get_sequence_logprob(image_path, prompt_text, target_text):
    async with semaphore:
        base64_image = await asyncio.to_thread(encode_image_to_base64, image_path)
        if not base64_image:
            print(f"[ERROR] 이미지 인코딩 실패: {image_path}")
            return -999.0

        try:
            response = await asyncio.wait_for(
                aclient.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                                {"type": "text", "text": prompt_text}
                            ]
                        },
                        {"role": "assistant", "content": target_text}
                    ],
                    max_tokens=2, 
                    logprobs=True,
                    top_logprobs=1,
                    extra_body={"echo": True} 
                ),
                timeout=120.0 
            )
            logprobs_data = response.choices[0].logprobs.content
            total_logprob = sum(t.logprob for t in logprobs_data if t.logprob is not None)
            return total_logprob / max(len(logprobs_data), 1)
        except asyncio.TimeoutError:
            print(f"[TIMEOUT] vLLM 타임아웃 (120초 초과)")
            return -999.0
        except Exception as e:
            print(f"[ERROR] API 호출 실패: {type(e).__name__}: {e}")
            return -999.0

def calculate_da_dpo_weight(lp_chosen, lp_rejected, beta=0.1):
    if lp_chosen == -999.0 or lp_rejected == -999.0:
        return 0.5
    reward_gap = beta * (lp_chosen - lp_rejected)
    try:
        weight = 1.0 - (1.0 / (1.0 + math.exp(-reward_gap)))
    except OverflowError:
        weight = 1.0 if reward_gap > 0 else 0.0
    return max(weight, 0.1)

# 통합된 행 처리 함수
async def process_row_with_idx(idx, row):
    lp_chosen = await get_sequence_logprob(row['image_path'], row['prompt'], row['chosen'])
    lp_rejected = await get_sequence_logprob(row['image_path'], row['prompt'], row['rejected'])
    weight = calculate_da_dpo_weight(lp_chosen, lp_rejected)
    return idx, lp_chosen, lp_rejected, weight

async def main():
    input_file = "dpo_pairs.csv"
    output_file = "dpo_pairs_with_weights.csv"

    if os.path.exists(output_file):
        print(f"기존 작업 내역을 불러옵니다...")
        dpo_df = pd.read_csv(output_file)
        print(f"[DEBUG] 로드된 행 수: {len(dpo_df)}")
        print(f"[DEBUG] 컬럼: {list(dpo_df.columns)}")
        if 'da_dpo_weight' in dpo_df.columns:
            print(f"[DEBUG] da_dpo_weight NaN 개수: {dpo_df['da_dpo_weight'].isna().sum()}")
            print(f"[DEBUG] da_dpo_weight 샘플 값: {dpo_df['da_dpo_weight'].head()}")
    else:
        print(f"새로운 데이터셋을 불러옵니다...")
        dpo_df = pd.read_csv(input_file)
        print(f"[DEBUG] 로드된 행 수: {len(dpo_df)}")
        for col in ['logprob_chosen', 'logprob_rejected', 'da_dpo_weight']:
            if col not in dpo_df.columns:
                dpo_df[col] = None
        print(f"[DEBUG] 컬럼 추가 완료: {list(dpo_df.columns)}")

    # 중요: 아직 처리되지 않은 행만 골라내기
    if 'da_dpo_weight' not in dpo_df.columns:
        print("[ERROR] 'da_dpo_weight' 컬럼이 없습니다!")
        return
        
    pending_mask = dpo_df['da_dpo_weight'].isna()
    pending_indices = dpo_df[pending_mask].index.tolist()
    
    print(f"[DEBUG] 처리 필요 행 수: {len(pending_indices)}")
    
    if not pending_indices:
        print("모든 작업이 완료되었습니다.")
        return

    print(f"총 {len(dpo_df)}개 중, 남은 {len(pending_indices)}개의 처리를 시작합니다.")
    print(f"[DEBUG] 첫 3개 인덱스: {pending_indices[:3]}")

    # 배치로 나누어 처리 (메모리 및 안정성 향상)
    BATCH_SIZE = 100
    completed_count = 0
    
    for batch_start in range(0, len(pending_indices), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(pending_indices))
        batch_indices = pending_indices[batch_start:batch_end]
        
        print(f"\n[배치] {batch_start+1}~{batch_end} / {len(pending_indices)} 처리 중...")
        
        # 배치 내에서 테스크 생성
        batch_tasks = []
        for idx in batch_indices:
            task = asyncio.create_task(process_row_with_idx(idx, dpo_df.loc[idx]))
            batch_tasks.append(task)
        
        # 배치 내 모든 작업 완료 대기
        batch_results = []
        with tqdm(total=len(batch_tasks), desc=f"배치 {batch_start//BATCH_SIZE + 1}", leave=False) as pbar:
            for coro in asyncio.as_completed(batch_tasks):
                try:
                    idx, lp_chosen, lp_rejected, weight = await coro
                    batch_results.append((idx, lp_chosen, lp_rejected, weight))
                    pbar.update(1)
                except Exception as e:
                    print(f"[ERROR] Task 처리 중 오류: {e}")
                    pbar.update(1)
        
        # 배치 결과 저장
        for idx, lp_chosen, lp_rejected, weight in batch_results:
            dpo_df.at[idx, 'logprob_chosen'] = lp_chosen
            dpo_df.at[idx, 'logprob_rejected'] = lp_rejected
            dpo_df.at[idx, 'da_dpo_weight'] = weight
            completed_count += 1
            print(f"  ✓ Row {idx}: chosen={lp_chosen:.4f}, rejected={lp_rejected:.4f}, weight={weight:.4f}")
        
        # 배치마다 파일에 저장
        print(f"  → 중간 저장: {completed_count}/{len(pending_indices)}")
        dpo_df.to_csv(output_file, index=False, encoding="utf-8-sig")

    # 최종 확인
    print(f"\n✅ 모든 작업 완료! '{output_file}'에 {completed_count}개 행 저장")
    print(f"   파일 크기: {os.path.getsize(output_file) / 1024 / 1024:.2f} MB")

if __name__ == "__main__":
    asyncio.run(main())