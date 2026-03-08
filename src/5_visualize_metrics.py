import os
import re
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# [설정] 학술 논문 스타일 그래프 세팅
# ==========================================
# 논문 가독성을 위해 폰트 크기와 스타일을 굵직하게 설정합니다.
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.dpi': 300, # 고해상도 출력
    'font.family': 'sans-serif' # 필요시 'Times New Roman' 등으로 변경
})

# ==========================================
# [경로 설정]
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_FIGURE_DIR = os.path.join(PROJECT_ROOT, "figures")

os.makedirs(OUTPUT_FIGURE_DIR, exist_ok=True)

RESULT_CSV_PATH = os.path.join(DATA_DIR, "eval_results_comparison.csv")

def extract_score_from_output(output_text):
    """
    새 모델이 생성한 텍스트(JSON 형태)에서 'd_answer_accuracy' 점수를 파싱합니다.
    """
    try:
        # JSON 블록 찾기 (```json ... ``` 형태 대비)
        match = re.search(r'\{.*\}', output_text, re.DOTALL)
        if match:
            json_str = match.group(0)
            parsed = json.loads(json_str)
            return float(parsed.get('d_answer_accuracy', 0))
    except Exception:
        pass
    return None # 파싱 실패 시 결측치 처리

def calculate_alignment(human_score, model_score, max_score=5.0):
    """
    인간 점수와 모델 점수의 일치율(Alignment)을 퍼센트(%)로 계산합니다.
    오차가 0이면 100%, 오차가 max_score면 0%입니다.
    $$ Alignment = (1 - \frac{|Human - Model|}{Max}) \times 100 $$
    """
    error = abs(human_score - model_score)
    return (1.0 - (error / max_score)) * 100.0

def main():
    print("📊 평가 결과 데이터를 불러오고 분석을 시작합니다...")
    
    if not os.path.exists(RESULT_CSV_PATH):
        print(f"❌ 오류: {RESULT_CSV_PATH} 파일이 없습니다. 4번 스크립트를 먼저 실행하세요.")
        return

    df = pd.read_csv(RESULT_CSV_PATH)

    # 1. 새 모델 텍스트에서 점수 파싱
    df['new_model_score'] = df['new_model_output'].apply(extract_score_from_output)
    
    # 결측치 제거 (파싱 실패한 데이터)
    df = df.dropna(subset=['human_score_accuracy', 'baseline_score_accuracy', 'new_model_score', 'difficulty'])

    # 2. Alignment Score 계산
    df['Baseline_Alignment'] = df.apply(lambda x: calculate_alignment(x['human_score_accuracy'], x['baseline_score_accuracy']), axis=1)
    df['HC_DA_DPO_Alignment'] = df.apply(lambda x: calculate_alignment(x['human_score_accuracy'], x['new_model_score']), axis=1)

    print(f"✅ 총 {len(df)}개의 유효한 데이터 분석 준비 완료.\n")

    # ==========================================
    # 📈 Figure 1: Overall Alignment Comparison
    # ==========================================
    plt.figure(figsize=(8, 6))
    overall_means = [df['Baseline_Alignment'].mean(), df['HC_DA_DPO_Alignment'].mean()]
    
    sns.barplot(
        x=['Baseline (Qwen-32B)', 'HC-DA-DPO (Ours)'], 
        y=overall_means, 
        palette=['#B0BEC5', '#1E88E5']
    )
    plt.title('Overall Human-Model Alignment Score (%)')
    plt.ylabel('Alignment Score (%)')
    plt.ylim(0, 100)
    
    # 막대 위에 점수 텍스트 표시
    for i, v in enumerate(overall_means):
        plt.text(i, v + 2, f"{v:.1f}%", ha='center', fontweight='bold')

    plt.tight_layout()
    fig1_path = os.path.join(OUTPUT_FIGURE_DIR, "fig1_overall_alignment.pdf")
    plt.savefig(fig1_path)
    print(f"저장 완료: {fig1_path}")

    # ==========================================
    # 📈 Figure 2: Performance by Difficulty (논문의 핵심!)
    # ==========================================
    plt.figure(figsize=(10, 6))
    
    # 데이터 형태 변환 (Melt) for Seaborn
    df_melt = pd.melt(
        df, 
        id_vars=['difficulty'], 
        value_vars=['Baseline_Alignment', 'HC_DA_DPO_Alignment'],
        var_name='Model', 
        value_name='Alignment'
    )
    df_melt['Model'] = df_melt['Model'].replace({'Baseline_Alignment': 'Baseline', 'HC_DA_DPO_Alignment': 'HC-DA-DPO'})
    
    # 난이도 순서 정렬을 위한 Categorical 변환 (예: 1~5점 척도라고 가정)
    df_melt['difficulty'] = pd.Categorical(df_melt['difficulty'], categories=sorted(df['difficulty'].unique()), ordered=True)

    sns.barplot(
        data=df_melt, 
        x='difficulty', 
        y='Alignment', 
        hue='Model', 
        palette=['#B0BEC5', '#FF8F00'],
        capsize=.1,
        errwidth=1.5
    )
    plt.title('Alignment Score by Human-Perceived Difficulty')
    plt.xlabel('Difficulty Level (Higher is Harder)')
    plt.ylabel('Alignment Score (%)')
    plt.legend(title='Model')
    plt.ylim(0, 100)

    plt.tight_layout()
    fig2_path = os.path.join(OUTPUT_FIGURE_DIR, "fig2_difficulty_breakdown.pdf")
    plt.savefig(fig2_path)
    print(f"저장 완료: {fig2_path}")

    # ==========================================
    # 📈 Figure 3: Absolute Error Distribution
    # ==========================================
    plt.figure(figsize=(8, 6))
    
    df['Baseline_Error'] = abs(df['human_score_accuracy'] - df['baseline_score_accuracy'])
    df['HC_DA_DPO_Error'] = abs(df['human_score_accuracy'] - df['new_model_score'])

    sns.kdeplot(df['Baseline_Error'], label='Baseline', fill=True, color='#B0BEC5', alpha=0.5)
    sns.kdeplot(df['HC_DA_DPO_Error'], label='HC-DA-DPO (Ours)', fill=True, color='#43A047', alpha=0.5)
    
    plt.title('Distribution of Absolute Errors against Human Score')
    plt.xlabel('Absolute Error (Lower is Better)')
    plt.ylabel('Density')
    plt.legend()
    
    plt.tight_layout()
    fig3_path = os.path.join(OUTPUT_FIGURE_DIR, "fig3_error_distribution.pdf")
    plt.savefig(fig3_path)
    print(f"저장 완료: {fig3_path}")
    print("\n🎉 모든 분석 및 시각화가 완료되었습니다! 논문에 바로 첨부하세요.")

if __name__ == "__main__":
    main()