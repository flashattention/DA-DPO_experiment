#!/bin/bash
# ==============================================================================
# 🚀 HC-DA-DPO Step 2: Weight Calculation Execution Script
# 🛠️ System Architect: Optimized for 4-GPU (A100 80GB) Multi-node Stability
# ==============================================================================

set -eou pipefail

# 1. VRAM 단편화 원천 차단
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# 2. CPU 스레드 병목 해소
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

# 3. [추가] HuggingFace Tokenizer 멀티프로세싱 데드락 방어
export TOKENIZERS_PARALLELISM=false

# 4. GPU 하드웨어 격리
export CUDA_VISIBLE_DEVICES="4,5,6,7"

WORK_DIR=$(pwd)
LOG_DIR="${WORK_DIR}/logs"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +'%Y%m%d_%H%M%S')
LOG_FILE="${LOG_DIR}/step2_weight_calc_${TIMESTAMP}.log"

# ==========================================
# 🌟 [자동 백그라운드 실행기 (Launcher)]
# ==========================================
if [ "${1:-}" != "bg" ]; then
    echo "========================================================================"
    echo "🚀 [STAGE 2] 가중치 계산을 백그라운드에서 시작합니다!"
    echo "💻 할당된 런처 PID: $$"
    echo "📝 로그 실시간 확인 명령어:"
    echo "tail -f ${LOG_FILE}"
    echo "========================================================================"

    nohup bash "$0" bg "${TIMESTAMP}" "${LOG_FILE}" > "${LOG_FILE}" 2>&1 &

    echo "✅ 백그라운드 실행 시작 완료 (프로세스 ID: $!)"
    exit 0
fi

TIMESTAMP="$2"
LOG_FILE="$3"

echo "========================================================================"
echo "🌟 [START] HC-DA-DPO Weight Calculation Pipeline"
echo "📅 Time: $(date)"
echo "💻 Allocated GPUs: ${CUDA_VISIBLE_DEVICES} (A100 80GB)"
echo "📝 Log File: ${LOG_FILE}"
echo "⚙️ Advanced Protections: ENABLED (VRAM Expandable, Tokenizer Safe)"
echo "========================================================================"

if python src/2_calc_dynamic_metrics.py; then
    echo "========================================================================"
    echo "✅ [SUCCESS] Weight calculation completed perfectly."
    echo "📅 End Time: $(date)"
    echo "========================================================================"
else
    EXIT_CODE=$?
    echo "========================================================================"
    echo "❌ [ERROR] Script failed with exit code ${EXIT_CODE}. Check ${LOG_FILE}"
    echo "📅 End Time: $(date)"
    echo "========================================================================"
    exit ${EXIT_CODE}
fi