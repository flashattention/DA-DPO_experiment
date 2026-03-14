import os
import json
import time
import argparse
import threading
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from openai import OpenAI, RateLimitError, APIError

# ==========================================
# [설정] 학술 논문 스타일 그래프 세팅
# ==========================================
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11, 
    'figure.dpi': 300, 
    'font.family': 'sans-serif',
    'pdf.fonttype': 42 # 논문 제출 시 폰트 임베딩 문제 방지
})

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# 파일 쓰기 충돌 방지를 위한 전역 Lock
file_lock = threading.Lock()

# ==========================================
# 🤖 [GPT 자동 채점 함수] Pairwise + Retry
# ==========================================
def call_openai_with_retry(client, judge_model, system_prompt, user_prompt, max_retries=5):
    """API 호출 및 지수 백오프(Exponential Backoff) 적용 재시도 로직"""
    base_delay = 2
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model=judge_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={ "type": "json_object" },
                temperature=0.0 # 평가의 일관성을 위해 0
            )
            return json.loads(completion.choices[0].message.content)
            
        except RateLimitError:
            delay = base_delay * (2 ** attempt)
            time.sleep(delay)
        except (APIError, Exception) as e:
            if attempt == max_retries - 1:
                return {"winner": "Error", "reason": str(e)}
            time.sleep(base_delay)
    return {"winner": "Error"}

def evaluate_pairwise_with_swap(client, judge_model, question, ground_truth, baseline_resp, target_resp):
    """
    Position Bias 상쇄를 위한 Swap Test 로직.
    1차: A(Baseline) vs B(Target)
    2차: A(Target) vs B(Baseline)
    결과가 일치할 때만 승리를 부여하고, 불일치 시 Tie 처리.
    """
    if pd.isna(baseline_resp) or str(baseline_resp).strip() == "":
        return "Target_Win" # Baseline이 빈칸이면 Target 승리
    if pd.isna(target_resp) or str(target_resp).strip() == "":
        return "Baseline_Win" # Target이 빈칸이면 Baseline 승리

    system_prompt = """You are an impartial, highly capable academic judge. 
Evaluate which of the two model responses is better aligned with the Ground Truth and better answers the User Question.
Consider accuracy, reasoning depth, and human-perceived clarity.
Output ONLY a JSON format: {"winner": "A" | "B" | "Tie", "reason": "concise explanation"}"""

    # --- 1차 평가 (A: Baseline, B: Target) ---
    prompt_1 = f"[Question]: {question}\n[Ground Truth]: {ground_truth}\n\n[Response A]: {baseline_resp}\n\n[Response B]: {target_resp}"
    res_1 = call_openai_with_retry(client, judge_model, system_prompt, prompt_1)
    winner_1 = res_1.get("winner", "Tie")

    # --- 2차 평가 Swap (A: Target, B: Baseline) ---
    prompt_2 = f"[Question]: {question}\n[Ground Truth]: {ground_truth}\n\n[Response A]: {target_resp}\n\n[Response B]: {baseline_resp}"
    res_2 = call_openai_with_retry(client, judge_model, system_prompt, prompt_2)
    winner_2 = res_2.get("winner", "Tie")

    # --- 결과 통합 (Consistency Check) ---
    if winner_1 == "B" and winner_2 == "A":
        return "Target_Win"
    elif winner_1 == "A" and winner_2 == "B":
        return "Baseline_Win"
    elif winner_1 == "Error" or winner_2 == "Error":
        return "Error"
    else:
        return "Tie" # 판단이 엇갈리거나 원래 Tie인 경우

# ==========================================
# ⚡ [병렬 처리 워커]
# ==========================================
def process_row(idx, row, client, judge_model, model_keys, cache_df, output_path):
    """단일 행(데이터)에 대한 모든 모델 비교 수행 및 실시간 저장"""
    question = row.get('question', 'N/A')
    ground_truth = row.get('ground_truth_answer', '')
    baseline_resp = row['Baseline_response']
    
    results = {"idx": idx}
    needs_update = False

    for target_model in model_keys:
        if target_model == "Baseline":
            continue
            
        col_name = f"Win_vs_{target_model}"
        
        # 이미 캐시(기존 결과)에 유효한 값이 있으면 API 호출 스킵 (비용 절약)
        if col_name in cache_df.columns and pd.notna(cache_df.loc[idx, col_name]) and cache_df.loc[idx, col_name] != "Error":
            results[col_name] = cache_df.loc[idx, col_name]
        else:
            # API 호출하여 평가
            target_resp = row[f"{target_model}_response"]
            winner = evaluate_pairwise_with_swap(client, judge_model, question, ground_truth, baseline_resp, target_resp)
            results[col_name] = winner
            needs_update = True
    
    # 새로운 평가 결과가 있다면 전체 DF 업데이트 후 파일 저장 (Thread-safe)
    if needs_update:
        with file_lock:
            for k, v in results.items():
                if k != "idx":
                    cache_df.at[idx, k] = v
            # 안전하게 임시 파일에 쓰고 덮어쓰기 (파일 손상 방지)
            temp_path = output_path + ".tmp"
            cache_df.to_csv(temp_path, index=False, encoding='utf-8-sig')
            os.replace(temp_path, output_path)

    return idx, results

# ==========================================
# 🚀 [메인 파이프라인]
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Top-Tier Pairwise Evaluation with Parallel Processing")
    parser.add_argument("--exp_dir", type=str, required=True, help="실험 폴더 경로 (예: runs/exp_...)")
    parser.add_argument("--judge_model", type=str, default="gpt-4o", help="평가 모델 (비용 절감을 원하면 gpt-4o-mini 권장)")
    parser.add_argument("--num_workers", type=int, default=10, help="병렬 처리 스레드 수 (권장 10~20)")
    args = parser.parse_args()

    EXP_DIR = os.path.join(PROJECT_ROOT, args.exp_dir)
    INPUT_CSV_PATH = os.path.join(EXP_DIR, "eval", "evaluation_results.csv")
    OUTPUT_CSV_PATH = os.path.join(EXP_DIR, "eval", "pairwise_evaluation_results.csv")
    OUTPUT_FIGURE_DIR = os.path.join(EXP_DIR, "figures")
    os.makedirs(OUTPUT_FIGURE_DIR, exist_ok=True)

    print(f"📊 [STAGE 3] Top-Tier Pairwise 평가를 시작합니다. (대상: {EXP_DIR})")
    print(f"⚙️  설정: Judge Model = {args.judge_model}, Workers = {args.num_workers}")

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY가 없습니다. 'export OPENAI_API_KEY=sk-...' 를 실행하세요.")
        return
    client = OpenAI(api_key=api_key)

    if not os.path.exists(INPUT_CSV_PATH):
        print(f"❌ 오류: {INPUT_CSV_PATH} 파일이 없습니다.")
        return

    # 원본 데이터 로드
    df = pd.read_csv(INPUT_CSV_PATH)
    
    # 이전에 중단된 작업이 있다면 이어서 진행하기 위해 로드
    if os.path.exists(OUTPUT_CSV_PATH):
        print("🔄 기존 평가 결과 파일을 발견했습니다. 이어서 작업을 진행합니다 (Resuming).")
        cache_df = pd.read_csv(OUTPUT_CSV_PATH)
        # 원본과 행 개수가 다르면 초기화
        if len(cache_df) != len(df):
            cache_df = df.copy()
    else:
        cache_df = df.copy()

    model_keys = ["Baseline", "Standard_DPO", "DA_DPO", "HC_DA_DPO"]
    display_names = ["Standard DPO", "DA-DPO", "HC-DA-DPO (Ours)"]

    # --- 1. 병렬 처리 (Thread Pool) 기반 API 평가 ---
    print("\n🚀 병렬 API 호출을 시작합니다. (진행률 바가 차오릅니다)")
    
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = [
            executor.submit(process_row, idx, row, client, args.judge_model, model_keys, cache_df, OUTPUT_CSV_PATH)
            for idx, row in df.iterrows()
        ]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating Pairs"):
            try:
                future.result() # 예외 발생 시 캐치
            except Exception as e:
                print(f"Thread error: {e}")

    print("\n✅ 모든 채점이 완료되었습니다. 최종 결과를 시각화합니다.\n")

    # --- 2. 평가 결과 집계 및 시각화 (논문용 Stacked Bar Chart) ---
    stats = {}
    target_models = [m for m in model_keys if m != "Baseline"]
    
    for target_model in target_models:
        col = f"Win_vs_{target_model}"
        total = len(cache_df)
        wins = len(cache_df[cache_df[col] == "Target_Win"])
        ties = len(cache_df[cache_df[col] == "Tie"])
        losses = len(cache_df[cache_df[col] == "Baseline_Win"])
        errors = len(cache_df[cache_df[col] == "Error"])
        
        valid_total = total - errors # 에러 난 항목 제외하고 비율 계산
        if valid_total == 0: valid_total = 1
        
        stats[target_model] = {
            "Win": (wins / valid_total) * 100,
            "Tie": (ties / valid_total) * 100,
            "Loss": (losses / valid_total) * 100
        }

    # 시각화 데이터 준비
    wins = [stats[m]["Win"] for m in target_models]
    ties = [stats[m]["Tie"] for m in target_models]
    losses = [stats[m]["Loss"] for m in target_models]

    plt.figure(figsize=(10, 6))
    bar_width = 0.6
    x = np.arange(len(display_names))

    # Stacked Bar 생성 (Loss -> Tie -> Win 순서로 쌓기)
    p1 = plt.bar(x, losses, bar_width, label='Loss (Baseline Better)', color='#EF9A9A', edgecolor='white')
    p2 = plt.bar(x, ties, bar_width, bottom=losses, label='Tie', color='#EEEEEE', edgecolor='white')
    p3 = plt.bar(x, wins, bar_width, bottom=np.array(losses)+np.array(ties), label='Win (Target Better)', color='#90CAF9', edgecolor='white')

    # 퍼센트 텍스트 중앙 삽입
    for i in range(len(x)):
        if losses[i] > 5: plt.text(x[i], losses[i]/2, f"{losses[i]:.1f}%", ha='center', va='center', color='black', fontsize=10)
        if ties[i] > 5: plt.text(x[i], losses[i] + ties[i]/2, f"{ties[i]:.1f}%", ha='center', va='center', color='black', fontsize=10)
        if wins[i] > 5: plt.text(x[i], losses[i] + ties[i] + wins[i]/2, f"{wins[i]:.1f}%", ha='center', va='center', color='black', fontweight='bold', fontsize=10)

    # 50% 기준선 (Baseline을 이기는 지점)
    plt.axhline(y=50, color='gray', linestyle='--', alpha=0.7)
    
    plt.title(f'Pairwise Win-Rate against Baseline (Judge: {args.judge_model})')
    plt.xticks(x, display_names, fontweight='bold')
    plt.ylabel('Percentage (%)')
    plt.ylim(0, 100)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1)) # 범례를 그래프 바깥으로
    
    plt.tight_layout()
    output_fig_path = os.path.join(OUTPUT_FIGURE_DIR, "fig1_pairwise_win_rate.pdf")
    plt.savefig(output_fig_path)
    
    print(f"🎉 논문용 시각화 완료! 그래프가 저장되었습니다: {output_fig_path}")
    print("================================================================")
    for i, model_name in enumerate(display_names):
        print(f"[{model_name}] Win: {wins[i]:.1f}% | Tie: {ties[i]:.1f}% | Loss: {losses[i]:.1f}%")
    print("================================================================")

if __name__ == "__main__":
    main()