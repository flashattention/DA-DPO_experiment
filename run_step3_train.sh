#!/bin/bash
set -e

# ==========================================
# 🌟 [개선] .env 파일이 있다면 자동으로 환경변수 싹 다 불러오기!
# ==========================================
if [ -f .env ]; then
    echo "✅ [.env] 파일에서 환경변수를 자동으로 불러옵니다."
    export $(cat .env | grep -v '^#' | xargs)
fi

# (쓸데없이 실행 막던 WANDB 검사 로직 삭제 완료!)

# ==========================================
# 🌟 [자동 백그라운드 실행기 (Launcher)] 🌟
# ==========================================
if [ "$1" != "bg" ]; then
    mkdir -p logs
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    LOG_FILE="logs/step3_train_${TIMESTAMP}.log"
    EXP_DIR="runs/exp_${TIMESTAMP}"
    
    echo "=================================================="
    echo "🚀 [STAGE 1] 모델 학습(Training)을 백그라운드에서 시작합니다!"
    echo "📁 실험 보관소: ${EXP_DIR}"
    echo "📝 로그 실시간 확인 명령어:"
    echo "tail -f ${LOG_FILE}"
    echo "=================================================="
    
    # 자기 자신을 nohup으로 다시 부르고 터미널을 놓아줍니다.
    nohup bash "$0" bg "${TIMESTAMP}" > "${LOG_FILE}" 2>&1 &
    
    echo "✅ 완벽하게 백그라운드에 던졌습니다! (프로세스 ID: $!)"
    echo "👉 이제 터미널 창을 닫거나 컴퓨터를 끄셔도 서버에서 끝까지 돌아갑니다."
    exit 0
fi

# ==========================================
# 🛠️ [실제 백그라운드 작업 영역] 🛠️
# ==========================================
TIMESTAMP=$2
EXP_DIR="runs/exp_${TIMESTAMP}"

mkdir -p "${EXP_DIR}"

CONFIGS=(
    "weight_dpo output_dpo_model DPO_Baseline_${TIMESTAMP}"
    "weight_da_dpo output_da_dpo_model DADPO_${TIMESTAMP}"
    "weight_hc_da_dpo output_hc_da_dpo_model HCDADPO_${TIMESTAMP}"
)

for config in "${CONFIGS[@]}"; do
    read -r WEIGHT_COL SUB_DIR RUN_NAME <<< "$config"
    FINAL_OUTPUT_DIR="${EXP_DIR}/${SUB_DIR}"
    
    echo "▶️ [학습 진행 중] 모델: ${RUN_NAME} -> ${FINAL_OUTPUT_DIR}"

    # 3번 파이썬 스크립트 실행
    torchrun --nproc_per_node=4 src/3_train.py \
        --weight_col "${WEIGHT_COL}" \
        --output_dir "${FINAL_OUTPUT_DIR}" \
        --run_name "${RUN_NAME}"
        
    echo "✅ ${RUN_NAME} 학습 종료! VRAM 회수를 위해 5초 대기..."
    sleep 5 
done

echo "=================================================="
echo "🎯 [STAGE 1 완료] 모든 모델 학습 완료!"
echo "👉 다음 단계(평가)를 위해 아래 명령어를 복사해서 실행하세요:"
echo "bash run_step4_evaluate.sh ${EXP_DIR}"
echo "=================================================="