#!/bin/bash
set -e

EXP_DIR=$1
NUM_GPUS=4

if [ -z "$EXP_DIR" ] || [ "$EXP_DIR" == "bg" ]; then
    echo "❌ [오류] 실험 폴더 경로가 필요합니다."
    echo "💡 사용법: bash run_step4_multi_gpu.sh runs/exp_20260309_113000"
    exit 1
fi

# ==========================================
# [백그라운드 실행 분기점]
# ==========================================
if [ "$2" != "bg" ]; then
    mkdir -p logs
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    LOG_FILE="logs/step4_evaluate_${TIMESTAMP}.log"
    
    echo "=================================================="
    echo "🚀 [STAGE 2] 4-way 분산 평가(Evaluation)를 백그라운드에서 시작합니다!"
    echo "📁 대상 실험 폴더: ${EXP_DIR}"
    echo "📝 통합 로그 확인: tail -f ${LOG_FILE}"
    echo "=================================================="
    
    # 자기 자신을 bg 인자와 함께 nohup으로 재실행
    nohup bash "$0" "${EXP_DIR}" bg > "${LOG_FILE}" 2>&1 &
    
    echo "✅ 성공적으로 백그라운드에 던졌습니다! (PID: $!)"
    exit 0
fi

# ==========================================
# [여기서부터는 실제 백그라운드(bg) 영역입니다]
# ==========================================
echo "▶️ [$(date)] 4번 파이썬 스크립트(4_evaluate_parallel.py) 4개 프로세스 동시 실행 중..."

# 4개의 프로세스를 백그라운드로 동시 실행
for i in $(seq 0 $((NUM_GPUS-1))); do
    echo "   ↳ GPU $i 에 프로세스 할당 중..."
    # 각 GPU마다 별도의 로그를 남기고 싶다면 아래 주석을 해제하세요. (현재는 통합 로그에 기록됨)
    # CUDA_VISIBLE_DEVICES=$i python src/4_evaluate_parallel.py --exp_dir "$EXP_DIR" --rank $i --world_size $NUM_GPUS > "logs/step4_gpu${i}_${TIMESTAMP}.log" 2>&1 &
    CUDA_VISIBLE_DEVICES=$i python src/4_evaluate.py \
        --exp_dir "$EXP_DIR" \
        --rank $i \
        --world_size $NUM_GPUS &
done

# 모든 백그라운드 프로세스(GPU 0~3)가 끝날 때까지 대기
wait

echo "=================================================="
echo "🎯 [$(date)] 모든 GPU 작업 완료! 결과물을 병합합니다."
echo "=================================================="

# 임시 분할 저장된 CSV 파일들을 하나로 병합하는 파이썬 스크립트 실행
python -c "
import pandas as pd, glob, os

exp_dir = '${EXP_DIR}'
files = glob.glob(f'{exp_dir}/eval/temp_res_gpu_*.csv')

if not files:
    print('❌ [오류] 병합할 결과 파일이 없습니다. 파이썬 에러 로그를 확인하세요.')
    exit(1)

print(f'✅ 찾은 임시 파일들: {files}')
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

out_path = f'{exp_dir}/eval/evaluation_results_4way.csv'
df.to_csv(out_path, index=False, encoding='utf-8-sig')

# 병합 완료 후 임시 파일 삭제
for f in files: 
    os.remove(f)

print(f'🎉 최종 결과물 생성 및 임시 파일 정리 완료: {out_path}')
"

echo "=================================================="
echo "🏁 [STAGE 2 완료] 전체 평가 추론 완벽 종료!"
echo "=================================================="