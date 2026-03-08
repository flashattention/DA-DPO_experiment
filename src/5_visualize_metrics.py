import os
import re
import json
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from openai import OpenAI # 🌟 GPT-4o 채점을 위한 라이브러리 추가

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
    'font.family': 'sans-serif' 
})

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# ==========================================
# 🤖 [GPT-4o 자동 채점 함수] LLM-as-a-Judge
# ==========================================
def evaluate_response_with_gpt(client, ground_truth, model_response):
    """
    GPT-4o를 사용하여 모델의 답변이 정답과 얼마나 일치하는지 0~5점으로 채점합니다.
    """
    # 답변이 비어있으면 0점 처리
    if pd.isna(model_response) or str(model_response).strip() == "":
        return 0.0

    system_prompt = "You are a strict academic grader. Compare the model's response to the ground truth. Score the alignment from 0.0 to 5.0, where 5.0 is perfectly correct and aligned, and 0.0 is completely wrong. Output ONLY a valid JSON format like: {\"score\": 4.5}"
    user_prompt = f"[Ground Truth]: {ground_truth}\n\n[Model Response]: {model_response}"

    try:
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={ "type": "json_object" },
            temperature=0.0 # 일관된 채점을 위해 0 설정
        )
        
        result_json = json.loads(completion.choices[0].message.content)
        return float(result_json.get("score", 0.0))
    except Exception as e:
        print(f"⚠️ API 채점 오류: {e}")
        return 0.0

def calculate_alignment(model_score, max_score=5.0):
    """5점 만점의 점수를 100점 만점의 Alignment 퍼센트(%)로 변환합니다."""
    return (model_score / max_score) * 100.0

def main():
    parser = argparse.ArgumentParser(description="Evaluate & Visualize with GPT-4o")
    parser.add_argument("--exp_dir", type=str, required=True, help="실험 폴더 경로 (예: runs/exp_...)")
    args = parser.parse_args()

    EXP_DIR = os.path.join(PROJECT_ROOT, args.exp_dir)
    INPUT_CSV_PATH = os.path.join(EXP_DIR, "eval", "evaluation_results.csv")
    OUTPUT_FIGURE_DIR = os.path.join(EXP_DIR, "figures")
    SCORED_CSV_PATH = os.path.join(EXP_DIR, "eval", "scored_evaluation_results.csv")
    
    os.makedirs(OUTPUT_FIGURE_DIR, exist_ok=True)

    print(f"📊 [STAGE 3] 채점 및 시각화를 시작합니다. (대상: {EXP_DIR})")
    
    if not os.path.exists(INPUT_CSV_PATH):
        print(f"❌ 오류: {INPUT_CSV_PATH} 파일이 존재하지 않습니다. 4번 스크립트를 먼저 실행하세요.")
        return

    df = pd.read_csv(INPUT_CSV_PATH)
    
    model_keys = ["Baseline", "Standard_DPO", "DA_DPO", "HC_DA_DPO"]
    display_names = ["Baseline", "Standard DPO", "DA-DPO", "HC-DA-DPO (Ours)"]
    colors = ['#B0BEC5', '#64B5F6', '#FFB300', '#1E88E5']

    # 🌟 OpenAI API 클라이언트 초기화
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("❌ 오류: OPENAI_API_KEY 환경변수가 설정되지 않았습니다. 터미널에 'export OPENAI_API_KEY=sk-...'를 입력하세요.")
        return
    client = OpenAI(api_key=api_key)

    # 1. GPT-4o를 이용한 4개 모델 순차 채점
    print("\n🤖 GPT-4o API를 호출하여 답변을 채점 중입니다. (비용이 발생할 수 있습니다)")
    for key in model_keys:
        response_col = f"{key}_response"
        score_col = f"{key}_score"
        
        if response_col in df.columns:
            tqdm.pandas(desc=f"Scoring {key}")
            # 정답(ground_truth_answer)과 모델 답변을 비교하여 점수 산출
            df[score_col] = df.apply(
                lambda row: evaluate_response_with_gpt(client, row['ground_truth_answer'], row[response_col]), 
                axis=1
            )
            # 100점 만점 %로 변환
            df[f"{key}_Alignment"] = df[score_col].apply(calculate_alignment)

    # 채점 결과 중간 저장
    df.to_csv(SCORED_CSV_PATH, index=False, encoding='utf-8-sig')
    print(f"✅ 채점 완료! 결과가 저장되었습니다: {SCORED_CSV_PATH}\n")

    # ==========================================
    # 📈 시각화 파트 (기존 로직 유지 및 경로 변경)
    # ==========================================
    print("📈 논문용 Figure 생성을 시작합니다...")
    
    # Figure 1: Overall Alignment Comparison
    plt.figure(figsize=(9, 6))
    overall_means = [df[f"{key}_Alignment"].mean() for key in model_keys]
    ax = sns.barplot(x=display_names, y=overall_means, palette=colors)
    plt.title('Overall Model-Ground Truth Alignment Score (%)')
    plt.ylabel('Alignment Score (%)')
    plt.ylim(0, 100)
    for i, v in enumerate(overall_means):
        plt.text(i, v + 1.5, f"{v:.1f}%", ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FIGURE_DIR, "fig1_overall_alignment.pdf"))

    # Figure 2: Performance by Difficulty
    if 'difficulty' in df.columns:
        plt.figure(figsize=(12, 6))
        value_vars = [f"{key}_Alignment" for key in model_keys]
        df_melt = pd.melt(df, id_vars=['difficulty'], value_vars=value_vars, var_name='Model', value_name='Alignment')
        name_mapping = {f"{k}_Alignment": name for k, name in zip(model_keys, display_names)}
        df_melt['Model'] = df_melt['Model'].map(name_mapping)
        
        # 난이도를 구간(Bin)으로 나누거나 카테고리화 (연속형 변수 방지)
        df_melt['difficulty_bin'] = pd.cut(df_melt['difficulty'], bins=3, labels=['Easy', 'Medium', 'Hard'])
        
        sns.barplot(data=df_melt, x='difficulty_bin', y='Alignment', hue='Model', palette=colors, capsize=.1, errwidth=1.5)
        plt.title('Alignment Score by Human-Perceived Difficulty')
        plt.xlabel('Difficulty Level (Based on Human Penalty)')
        plt.ylabel('Alignment Score (%)')
        plt.legend(title='Model')
        plt.ylim(0, 100)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_FIGURE_DIR, "fig2_difficulty_breakdown.pdf"))

    print(f"🎉 모든 분석 및 시각화가 완료되었습니다! 논문 첨부용 PDF 확인: {OUTPUT_FIGURE_DIR}")

if __name__ == "__main__":
    main()