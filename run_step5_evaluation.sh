#!/bin/bash
set -e

# ==========================================
# ⚙️ [설정] 평가 파라미터 
# ==========================================
JUDGE_MODEL="gpt-4o" 
NUM_WORKERS=10            
PYTHON_SCRIPT="src/5_run_evaluation.py" 

EXP_DIR=$1

if [ -z "$EXP_DIR" ] || [ "$EXP_DIR" == "bg" ]; then
    echo "❌ [오류] 실험 폴더 경로가 필요합니다."
    echo "💡 사용법: bash $0 runs/exp_YOUR_EXP_ID"
    exit 1
fi

# ==========================================
# 🔐 [.env 파일 자동 로드]
# ==========================================
if [ -f ".env" ]; then
    echo "🔍 [.env] 파일을 발견하여 환경변수를 로드합니다..."
    set -a
    source .env
    set +a
else
    echo "⚠️ [.env] 파일을 찾을 수 없습니다. 시스템 환경변수를 확인합니다."
fi

if [ -z "$OPENAI_API_KEY" ]; then
    echo "❌ [오류] OPENAI_API_KEY가 설정되지 않았습니다."
    exit 1
fi

# ==========================================
# 🚀 [포그라운드(Foreground) - 백그라운드 분기]
# ==========================================
if [ "$2" != "bg" ]; then
    mkdir -p logs
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    LOG_FILE="logs/step5_eval_${TIMESTAMP}.log"
    
    echo "================================================================"
    echo "📊 [STAGE 5] Top-Tier Pairwise Evaluation 파이프라인 시작"
    echo "📁 대상 실험 폴더: ${EXP_DIR}"
    echo "================================================================"
    
    nohup bash "$0" "${EXP_DIR}" bg > "${LOG_FILE}" 2>&1 &
    
    echo "✅ 성공적으로 백그라운드에 던졌습니다! 실시간 로그를 확인하세요:"
    echo "👉 tail -f ${LOG_FILE}"
    exit 0
fi

# ==========================================
# ⚙️ [백그라운드(Background) - 실제 작업 수행 영역]
# ==========================================
EVAL_DIR="${EXP_DIR}/eval"
MERGED_CSV="${EVAL_DIR}/evaluation_results.csv"

# 1️⃣ 4개의 GPU 분할 CSV 파일을 하나로 병합 (Python One-liner)
if [ ! -f "$MERGED_CSV" ]; then
    echo "🔄 분할된 GPU 추론 결과(temp_res_gpu_*.csv)를 병합합니다..."
    python -c "
import pandas as pd
import glob
import os

eval_dir = '${EVAL_DIR}'
merged_csv = '${MERGED_CSV}'
files = glob.glob(os.path.join(eval_dir, 'temp_res_gpu_*.csv'))

if files:
    print(f'   -> 발견된 파일 {len(files)}개 병합 중...')
    df_list = [pd.read_csv(f) for f in files]
    merged_df = pd.concat(df_list, ignore_index=True)
    merged_df.to_csv(merged_csv, index=False, encoding='utf-8-sig')
    print('   -> 병합 완료! evaluation_results.csv 생성됨.')
else:
    print('❌ 병합할 temp_res_gpu_*.csv 파일이 없습니다. 경로를 확인해주세요.')
    exit(1)
"
else:
    echo "✅ 병합된 evaluation_results.csv 파일이 이미 존재합니다."
fi

# 2️⃣ 본 평가 스크립트 실행
echo "▶️ [$(date +"%Y-%m-%d %H:%M:%S")] GPT-4o 채점 스크립트(${PYTHON_SCRIPT}) 실행 중..."

python ${PYTHON_SCRIPT} \
    --exp_dir "${EXP_DIR}" \
    --judge_model "${JUDGE_MODEL}" \
    --num_workers ${NUM_WORKERS}

echo "================================================================"
echo "🎉 [ALL DONE] 대성공! 병렬 상대평가 및 시각화가 무사히 종료되었습니다."
echo "📂 [최종 결과물 확인]: ${EXP_DIR}/figures/"
echo "================================================================"