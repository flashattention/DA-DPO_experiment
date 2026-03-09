#!/bin/bash
set -e

EXP_DIR=$1

if [ -z "$EXP_DIR" ] || [ "$EXP_DIR" == "bg" ]; then
    echo "❌ [오류] 실험 폴더 경로가 필요합니다."
    echo "💡 사용법: bash run_step5_visualize.sh runs/exp_20260309_113000"
    exit 1
fi

if [ -z "$OPENAI_API_KEY" ]; then
    echo "❌ [오류] OPENAI_API_KEY 환경변수가 설정되지 않았습니다."
    exit 1
fi

# 백그라운드 실행 분기점
if [ "$2" != "bg" ]; then
    mkdir -p logs
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    LOG_FILE="logs/step5_visualize_${TIMESTAMP}.log"
    
    echo "=================================================="
    echo "📊 [STAGE 3] GPT-4o 채점 및 시각화를 백그라운드에서 시작합니다!"
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
echo "▶️ 5번 파이썬 스크립트(5_visualize.py) 실행 중..."
python 5_visualize.py --exp_dir "${EXP_DIR}"

echo "=================================================="
echo "🎉 [ALL DONE] 대성공! 시각화 파이프라인이 무사히 종료되었습니다."
echo "📂 [최종 결과물 확인]: ${EXP_DIR}/figures/"
echo "=================================================="