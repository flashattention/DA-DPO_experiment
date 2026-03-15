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
    """Position Bias 상쇄를 위한 Swap Test 로직 (1차, 2차 크로스 체크)"""
    # 응답이 비어있을 경우에 대한 방어
    if not isinstance(baseline_resp, str) or not baseline_resp.strip(): 
        return "Target_Win"
    if not isinstance(target_resp, str) or not target_resp.strip(): 
        return "Baseline_Win"

    system_prompt = """You are an impartial, highly capable academic judge evaluating JSON outputs. 
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
        return "Tie"

# ==========================================
# ⚡ [병렬 처리 워커]
# ==========================================
def process_row(idx, row, client, judge_model, model_keys, df_shared, output_path):
    """단일 행(데이터)에 대한 모든 모델 비교 수행 및 실시간 저장"""
    # 4번 코드 구조에 맞게 질문 열(keyword) 매핑
    question = row.get('keyword', 'N/A')
    ground_truth = row.get('ground_truth_answer', '')
    
    # 🚨 핵심 수정: 4번 추론 코드가 내뱉은 "Detailed_JSON" 컬럼으로 매핑
    baseline_col = "Baseline_Detailed_JSON"
    baseline_resp = row.get(baseline_col, "{}")
    
    results = {"idx": idx}
    needs_update = False

    for target_model in model_keys:
        if target_model == "Baseline":
            continue
            
        col_name = f"Win_vs_{target_model}"
        target_col = f"{target_model}_Detailed_JSON"
        
        # 이미 평가된 항목은 건너뛰기
        with file_lock:
            existing_val = df_shared.at[idx, col_name] if col_name in df_shared.columns else None

        if pd.notna(existing_val) and existing_val != "Error":
            results[col_name] = existing_val
        else:
            # API 호출
            target_resp = row.get(target_col, "{}")
            winner = evaluate_pairwise_with_swap(client, judge_model, question, ground_truth, baseline_resp, target_resp)
            results[col_name] = winner
            needs_update = True
    
    if needs_update:
        with file_lock:
            for k, v in results.items():
                if k != "idx":
                    df_shared.at[idx, k] = v
            # 안전한 저장 (중간에 끊겨도 파일이 날아가지 않도록 임시 파일 사용)
            temp_path = output_path + ".tmp"
            df_shared.to_csv(temp_path, index=False, encoding='utf-8-sig')
            os.replace(temp_path, output_path)

    return idx, results

# ==========================================
# 🚀 [메인 파이프라인]
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, required=True)
    parser.add_argument("--judge_model", type=str, default="gpt-4o")
    parser.add_argument("--num_workers", type=int, default=10)
    args = parser.parse_args()

    EXP_DIR = args.exp_dir
    INPUT_CSV_PATH = os.path.join(EXP_DIR, "eval", "evaluation_results.csv")
    OUTPUT_CSV_PATH = os.path.join(EXP_DIR, "eval", "pairwise_evaluation_results.csv")
    OUTPUT_FIGURE_DIR = os.path.join(EXP_DIR, "figures")
    os.makedirs(OUTPUT_FIGURE_DIR, exist_ok=True)

    print(f"📊 [STAGE 5] Top-Tier Pairwise 평가를 시작합니다. (대상: {EXP_DIR})")
    print(f"⚙️  설정: Judge Model = {args.judge_model}, Workers = {args.num_workers}")

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    if not os.path.exists(INPUT_CSV_PATH):
        print(f"❌ 오류: {INPUT_CSV_PATH} 파일이 없습니다.")
        return

    df = pd.read_csv(INPUT_CSV_PATH)
    
    # 캐시 로드 또는 초기화
    if os.path.exists(OUTPUT_CSV_PATH):
        print("🔄 기존 평가 결과(캐시)를 발견했습니다. 이어서 진행합니다.")
        cache_df = pd.read_csv(OUTPUT_CSV_PATH)
        if len(cache_df) != len(df):
            cache_df = df.copy()
    else:
        cache_df = df.copy()

    model_keys = ["Baseline", "Standard_DPO", "DA_DPO", "HC_DA_DPO"]
    display_names = ["Standard DPO", "DA-DPO", "HC-DA-DPO (Ours)"]

    # 캐시용 빈 컬럼 생성
    for m in model_keys:
        if m != "Baseline":
            col = f"Win_vs_{m}"
            if col not in cache_df.columns:
                cache_df[col] = None

    print("\n🚀 병렬 API 호출을 시작합니다. (진행률 바가 차오릅니다)")
    
    tasks = []
    # 평가가 안 된 항목만 추출하여 tasks에 추가 (Resume 기능)
    for idx, row in cache_df.iterrows():
        needs_eval = False
        for m in model_keys:
            if m != "Baseline" and pd.isna(row.get(f"Win_vs_{m}")):
                needs_eval = True
                break
        if needs_eval:
            tasks.append((idx, row))

    if tasks:
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            futures = [
                executor.submit(process_row, task[0], task[1], client, args.judge_model, model_keys, cache_df, OUTPUT_CSV_PATH)
                for task in tasks
            ]
            for _ in tqdm(as_completed(futures), total=len(futures), desc="Evaluating Pairs"):
                pass
    else:
        print("✅ 모든 항목이 이미 평가되었습니다. 바로 시각화를 진행합니다.")

    print("\n✅ 모든 채점이 완료되었습니다. 최종 결과를 시각화합니다.\n")

    # ==========================================
    # 📊 [시각화] 논문용 Stacked Bar Chart
    # ==========================================
    stats = {}
    target_models = [m for m in model_keys if m != "Baseline"]
    
    for target_model in target_models:
        col = f"Win_vs_{target_model}"
        wins = len(cache_df[cache_df[col] == "Target_Win"])
        ties = len(cache_df[cache_df[col] == "Tie"])
        losses = len(cache_df[cache_df[col] == "Baseline_Win"])
        errors = len(cache_df[cache_df[col] == "Error"])
        
        valid_total = len(cache_df) - errors
        if valid_total == 0: valid_total = 1
        
        stats[target_model] = {
            "Win": (wins / valid_total) * 100,
            "Tie": (ties / valid_total) * 100,
            "Loss": (losses / valid_total) * 100
        }

    wins = [stats[m]["Win"] for m in target_models]
    ties = [stats[m]["Tie"] for m in target_models]
    losses = [stats[m]["Loss"] for m in target_models]

    plt.figure(figsize=(10, 6))
    bar_width = 0.6
    x = np.arange(len(display_names))

    p1 = plt.bar(x, losses, bar_width, label='Loss (Baseline Better)', color='#EF9A9A', edgecolor='white')
    p2 = plt.bar(x, ties, bar_width, bottom=losses, label='Tie', color='#EEEEEE', edgecolor='white')
    p3 = plt.bar(x, wins, bar_width, bottom=np.array(losses)+np.array(ties), label='Win (Target Better)', color='#90CAF9', edgecolor='white')

    for i in range(len(x)):
        if losses[i] > 5: plt.text(x[i], losses[i]/2, f"{losses[i]:.1f}%", ha='center', va='center', color='black', fontsize=10)
        if ties[i] > 5: plt.text(x[i], losses[i] + ties[i]/2, f"{ties[i]:.1f}%", ha='center', va='center', color='black', fontsize=10)
        if wins[i] > 5: plt.text(x[i], losses[i] + ties[i] + wins[i]/2, f"{wins[i]:.1f}%", ha='center', va='center', color='black', fontweight='bold', fontsize=10)

    plt.axhline(y=50, color='gray', linestyle='--', alpha=0.7)
    plt.title(f'Pairwise Win-Rate against Baseline (Judge: {args.judge_model})', pad=20, fontweight='bold')
    plt.xticks(x, display_names, fontweight='bold')
    plt.ylabel('Percentage (%)', fontweight='bold')
    plt.ylim(0, 100)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    
    plt.tight_layout()
    output_fig_path = os.path.join(OUTPUT_FIGURE_DIR, "fig1_pairwise_win_rate_detailed.pdf")
    plt.savefig(output_fig_path)
    
    print(f"🎉 논문용 시각화 완료! 그래프가 저장되었습니다: {output_fig_path}")
    print("================================================================")
    for i, model_name in enumerate(display_names):
        print(f"[{model_name}] Win: {wins[i]:.1f}% | Tie: {ties[i]:.1f}% | Loss: {losses[i]:.1f}%")
    print("================================================================")

if __name__ == "__main__":
    main()