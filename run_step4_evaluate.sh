#!/bin/bash
set -e

EXP_DIR=$1

if [ -z "$EXP_DIR" ] || [ "$EXP_DIR" == "bg" ]; then
    echo "❌ [오류] 실험 폴더 경로가 필요합니다."
    echo "💡 사용법: bash run_step4_evaluate.sh runs/exp_20260309_113000"
    exit 1
fi

# 백그라운드 실행 분기점
if [ "$2" != "bg" ]; then
    mkdir -p logs
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    LOG_FILE="logs/step4_evaluate_${TIMESTAMP}.log"
    
    echo "=================================================="
    echo "🚀 [STAGE 2] 8-way 평가(Evaluation)를 백그라운드에서 시작합니다!"
    echo "📁 대상 실험 폴더: ${EXP_DIR}"
    echo "📝 로그 확인: tail -f ${LOG_FILE}"
    echo "=================================================="
    
    nohup bash "$0" "${EXP_DIR}" bg > "${LOG_FILE}" 2>&1 &
    
    echo "✅ 성공적으로 백그라운드에 던졌습니다! (PID: $!)"
    exit 0
fi

# ==========================================
# [여기서부터는 실제 백그라운드(bg) 영역입니다]
# ==========================================
echo "▶️ 4번 파이썬 스크립트(4_evaluate.py) 실행 중..."
python 4_evaluate.py --exp_dir "${EXP_DIR}"

echo "=================================================="
echo "🎯 [STAGE 2 완료] 8-way 평가 추론 완벽 종료!"
echo "=================================================="