#!/bin/bash
set -e

# ==========================================
# 🌟 [개선] .env 환경변수 로드
# ==========================================
if [ -f .env ]; then
    echo "✅ [.env] 파일에서 환경변수를 자동으로 불러옵니다."
    export $(cat .env | grep -v '^#' | xargs)
fi

# ==========================================
# 🌟 [자동 백그라운드 실행기 (Launcher)]
# ==========================================
if [ "$1" != "bg" ]; then
    mkdir -p logs
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    LOG_FILE="logs/step3_train_${TIMESTAMP}.log"
    EXP_DIR="runs/exp_${TIMESTAMP}"
    
    echo "=================================================="
    echo "🚀 [STAGE 1] 모델 분산 학습(Training)을 백그라운드에서 시작합니다!"
    echo "💻 할당된 런처 PID: $$"
    echo "📁 실험 보관소: ${EXP_DIR}"
    echo "📝 로그 실시간 확인 명령어:"
    echo "tail -f ${LOG_FILE}"
    echo "=================================================="
    
    nohup bash "$0" bg "${TIMESTAMP}" > "${LOG_FILE}" 2>&1 &
    
    echo "✅ 완벽하게 백그라운드에 던졌습니다! (프로세스 ID: $!)"
    echo "👉 이제 터미널 창을 닫아도 학습은 계속됩니다."
    exit 0
fi

# ==========================================
# 🛠️ [실제 백그라운드 작업 영역 - System Architect Settings] 🛠️
# ==========================================
TIMESTAMP=$2
EXP_DIR="runs/exp_${TIMESTAMP}"
mkdir -p "${EXP_DIR}"

# 🚨 [수석의 Deep Fix 1] 공용 서버를 위한 자원/통신 방어벽
# 4개의 GPU를 풀로 사용 (2번 스크립트에서 1장만 썼다면, 여기선 4장 모두 할당)
export CUDA_VISIBLE_DEVICES="4,5,6,7" # 상황에 맞게 수정
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# 🚨 [수석의 Deep Fix 2] 거대 모델 분산 학습 (NCCL) 타임아웃 & 데드락 방지
export TORCH_DISTRIBUTED_DEBUG=INFO
export NCCL_DEBUG=WARN
export NCCL_TIMEOUT=7200 # 타임아웃 2시간으로 연장 (초기 모델 로딩 지연 대비)

# ==========================================
# 🌟 7가지 전체 Ablation Configurations
# ==========================================
CONFIGS=(
    "weight_dpo output_dpo_model DPO_Baseline_${TIMESTAMP}"
    "weight_da_dpo output_da_dpo_model DADPO_${TIMESTAMP}"
    "weight_hc_dpo output_hc_dpo_model HCDPO_${TIMESTAMP}"
    "weight_hc_da_dpo output_hc_da_dpo_model HCDADPO_${TIMESTAMP}"
    "weight_hc_da_dpo_opt1 output_hc_da_dpo_opt1_model HCDADPO_OPT1_${TIMESTAMP}"
    "weight_hc_da_dpo_opt3 output_hc_da_dpo_opt3_model HCDADPO_OPT3_${TIMESTAMP}"
    "weight_hc_da_dpo_opt1_3 output_hc_da_dpo_opt1_3_model HCDADPO_OPT1_3_${TIMESTAMP}"
)

for config in "${CONFIGS[@]}"; do
    read -r WEIGHT_COL SUB_DIR RUN_NAME <<< "$config"
    FINAL_OUTPUT_DIR="${EXP_DIR}/${SUB_DIR}"
    
    # 🚨 [수석의 Deep Fix 3] 포트 충돌 방지를 위한 동적 마스터 포트 할당
    # 29500 ~ 29600 사이의 랜덤 포트를 생성하여 공용 서버 충돌 원천 차단
    RANDOM_PORT=$(( RANDOM % 100 + 29500 ))
    
    echo "=================================================="
    echo "▶️ [학습 시작] 모델: ${RUN_NAME}"
    echo "▶️ 저장 경로: ${FINAL_OUTPUT_DIR}"
    echo "▶️ 사용 포트: ${RANDOM_PORT}"
    echo "=================================================="

    # 분산 학습 실행
    torchrun --nproc_per_node=4 --master_port=${RANDOM_PORT} src/3_train.py \
        --weight_col "${WEIGHT_COL}" \
        --output_dir "${FINAL_OUTPUT_DIR}" \
        --run_name "${RUN_NAME}"
        
    # 🚨 [수석의 Deep Fix 4] 확실한 VRAM 회수 (VRAM Hangover 방지)
    echo "✅ ${RUN_NAME} 학습 종료! VRAM의 완벽한 반환을 위해 20초간 쿨다운을 진행합니다..."
    sync # OS 캐시 디스크 동기화
    sleep 20 
done

echo "=================================================="
echo "🎯 모든 실험군(7종) 모델 학습 완료!"
echo "=================================================="