import pandas as pd
import torch
import torch.nn as nn
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration
from peft import LoraConfig, get_peft_model
from trl import DPOTrainer, DPOConfig
from typing import Dict, Union, Any

# 1. 데이터 불러오기 (가중치 포함)
df = pd.read_csv("dpo_pairs_with_weights.csv")

# 2. TRL DPOTrainer가 요구하는 기본 컬럼명으로 매핑
# (prompt, chosen, rejected 컬럼이 필수적으로 있어야 합니다)
# da_dpo_weight 컬럼을 weight로 매핑
df = df.rename(columns={'da_dpo_weight': 'weight'})
dataset = Dataset.from_pandas(df)

# 학습/검증 데이터셋 분리 (예: 90% 학습, 10% 검증)
dataset = dataset.train_test_split(test_size=0.1)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

model_id = "Qwen/Qwen3-VL-32B-Instruct"

# 4-bit 양자화 설정
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

# 모델 및 프로세서 로드
model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_id, 
    quantization_config=bnb_config, 
    device_map="auto",  # CUDA_VISIBLE_DEVICES로 GPU 제어
    dtype=torch.bfloat16,
    low_cpu_mem_usage=True,  # 이 옵션을 추가하세요!
    trust_remote_code=True
)
processor = AutoProcessor.from_pretrained(model_id)

# QLoRA 설정 (주의: 멀티모달 모델에 맞게 target_modules 지정 필요)
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# 학습 설정
training_args = DPOConfig(
    output_dir="./results",
    beta=0.1,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=5e-5,
    weight_decay=0.01,
    warmup_steps=100,
    logging_steps=10,
    save_steps=500,
    eval_strategy="steps",
    eval_steps=500,
    save_total_limit=3,
    load_best_model_at_end=True,
    fp16=False,  # bf16 사용하므로 fp16 비활성화
    bf16=True,   # A100 계열은 bf16이 훨씬 성능이 좋음
    tf32=True,   # Ampere 아키텍처 가속
    gradient_checkpointing=True,  # 메모리 절약 필수 (32B 모델)
    dataloader_num_workers=0,     # 멀티모달 데이터로 인한 호환성 문제 방지
    remove_unused_columns=False,  # 가중치(weight) 컬럼이 사라지지 않게 함
)

class DADPOTrainer(DPOTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        """
        배치 데이터(inputs)에 포함된 'weight'를 직접 꺼내서 
        DPO Loss에 곱해주는 방식으로 수정
        """
       # 1. 모델에 들어가면 안 되는 커스텀 인자(weight)를 미리 빼냅니다.
        weights = inputs.pop("weight", None)
        
        # 2. 부모 클래스(DPOTrainer)의 원본 compute_loss를 안전하게 호출합니다.
        # 내부 메트릭도 함께 받아오기 위해 return_outputs=True로 강제 지정합니다.
        outputs = super().compute_loss(
            model, 
            inputs, 
            return_outputs=True, 
            num_items_in_batch=num_items_in_batch, 
            **kwargs
        )
        loss, metrics = outputs
        
        # 3. 빼두었던 난이도 가중치를 로스에 적용합니다.
        if weights is not None:
            # 에러 방지: 가중치 텐서를 loss가 계산된 동일한 GPU로 이동시킵니다.
            weights = weights.to(loss.device) 
            loss = loss * weights.mean()
            
        # 4. 원래 사용자가 요청한 형태에 맞춰 반환합니다.
        return (loss, metrics) if return_outputs else loss

# Trainer 실행
trainer = DADPOTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

print("학습 시작...")
trainer.train()
print("학습 완료! 모델 저장 중...")
trainer.save_model("qwen3-vl-da-dpo-scoring-adapter")
print("모델 저장 완료!")