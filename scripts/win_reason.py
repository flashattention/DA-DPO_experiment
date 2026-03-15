import pandas as pd
import json

df = pd.read_csv("runs/exp_20260309_123727/eval/pairwise_evaluation_results.csv")
win_cases = df[df['Win_vs_DA_DPO'] == 'Target_Win']

for i, (idx, row) in enumerate(win_cases.head(3).iterrows()):
    print(f"📌 Case {i+1} (Index: {idx}) - Keyword: {row['keyword']}")
    
    # JSON 파싱
    hc_json = json.loads(row['HC_DA_DPO_Detailed_JSON'])
    da_json = json.loads(row['DA_DPO_Detailed_JSON'])
    
    print(f"✅ HC (Ours) English: {hc_json.get('a3_keyword_english')}")
    print(f"❌ DA-DPO English: {da_json.get('a3_keyword_english')}")
    
    # 만약 'reason'이나 'comment' 필드가 있다면 출력
    print(f"💬 HC Score: {hc_json.get('a1_keyword_cultural')} / DA Score: {da_json.get('a1_keyword_cultural')}")
    print("-" * 60)