import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from peft import PeftModel

# 경로 설정
base_model_id = "Qwen/Qwen3-VL-32B-Instruct"
adapter_path = "./qwen3-vl-da-dpo-scoring-adapter" # Step 3에서 저장한 LoRA 폴더
save_path = "./qwen3-vl-da-dpo-merged"             # 최종 병합 모델이 저장될 폴더

# 1. Base 모델을 bfloat16 정밀도로 로드 (4-bit로 로드하지 않음에 주의!)
print("Base 모델 로드 중...")
base_model = Qwen3VLForConditionalGeneration.from_pretrained(
    base_model_id,
    dtype=torch.bfloat16,
    device_map="auto"
)
processor = AutoProcessor.from_pretrained(base_model_id)

# 2. Base 모델에 LoRA 어댑터 결합
print("LoRA 어댑터 불러오는 중...")
model = PeftModel.from_pretrained(base_model, adapter_path)

# 3. 가중치 영구 병합 (Merge and Unload)
# 원본 가중치(W)에 LoRA 가중치(ΔW)를 더하여 W' = W + ΔW 로 만듭니다.
print("가중치 병합 중 (시간이 다소 소요됩니다)...")
merged_model = model.merge_and_unload()

# 4. 최종 모델 저장
print(f"병합된 모델을 '{save_path}' 에 분할 저장합니다...")
merged_model.save_pretrained(
    save_path, 
    safe_serialization=True,   # 안전하고 로딩이 빠른 safetensors 포맷 사용
    max_shard_size="5GB"       # 60GB 모델을 5GB씩 쪼개서 저장 (관리/업로드 용이)
)
processor.save_pretrained(save_path)
print("성공적으로 병합 및 저장이 완료되었습니다!")