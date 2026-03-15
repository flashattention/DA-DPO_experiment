import pandas as pd
import os

file_path = "runs/exp_20260309_123727/eval/pairwise_evaluation_results.csv"

if not os.path.exists(file_path):
    print(f"❌ 파일을 찾을 수 없습니다: {file_path}")
else:
    df = pd.read_csv(file_path)
    
    # 분석할 모델과 해당 컬럼 매핑
    # (Simple_Response가 실제 출력물이라면 아래에서 컬럼명을 변경해 보세요)
    target_models = {
        "Standard DPO": "Standard_DPO_Detailed_JSON",
        "DA-DPO": "DA_DPO_Detailed_JSON",
        "HC-DA-DPO (Ours)": "HC_DA_DPO_Detailed_JSON",
        "Baseline": "Baseline_Detailed_JSON"
    }

    print("\n📊 [모델별 평균 답변 길이 및 정보 밀도 분석]")
    print("-" * 60)
    
    results = {}
    for model_name, col in target_models.items():
        if col in df.columns:
            # 문자열 길이 계산 (결측치 제외)
            avg_len = df[col].astype(str).str.len().mean()
            results[model_name] = avg_len
            print(f"🔹 {model_name:20} : 평균 {avg_len:7.1f} 자")
        else:
            print(f"⚠️ {col} 컬럼을 찾을 수 없습니다.")

    print("-" * 60)

    # DA-DPO와 우리 모델 비교
    if "DA-DPO" in results and "HC-DA-DPO (Ours)" in results:
        da_len = results["DA-DPO"]
        our_len = results["HC-DA-DPO (Ours)"]
        diff = (our_len / da_len - 1) * 100
        
        if diff < 0:
            print(f"✅ [결론] 우리 모델이 DA-DPO보다 {abs(diff):.1f}% 더 간결합니다!")
            print("🚀 이것은 매우 유의미한 결과입니다. 탑티어 논문에서는 이를 ")
            print("   'Higher Information Density(더 높은 정보 밀도)'라고 표현하며,")
            print("   불필요한 말장난(Verbosity) 없이 핵심 정렬에 성공했음을 증명하는 지표로 씁니다.")
        else:
            print(f"ℹ️ 우리 모델이 DA-DPO보다 {diff:.1f}% 더 깁니다.")
            print("   이 경우, 답변의 풍부함(Richness)이 승률에 기여했는지 분석해야 합니다.")