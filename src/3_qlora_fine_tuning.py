import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import DPOTrainer
import torch.nn.functional as F

# ==========================================
# [경로 자동 설정]
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output_da_dpo_model")

DATASET_PATH = os.path.join(DATA_DIR, "train_hc_dpo_pairs.jsonl")

# ==========================================
# [설정] 모델 및 학습 하이퍼파라미터
# ==========================================
MODEL_ID = "Qwen/Qwen3-VL-32B-Thinking"

# ==========================================
# 🛠️ 핵심 기여점: Custom DA-DPO Trainer
# ==========================================
class DifficultyAwareDPOTrainer(DPOTrainer):
    """
    일반 DPO Loss에 각 데이터 샘플의 'difficulty_weight'를 곱하여
    난이도 기반 학습(DA-DPO)을 수행하는 커스텀 트레이너입니다.
    """
    def get_batch_loss_metrics(self, model, batch, train_eval="train"):
        """
        trl 라이브러리의 DPO 내부 로직을 오버라이딩하여 가중치를 적용합니다.
        """
        # 1. 기본 DPO 로직을 통해 policy와 reference의 logprob을 가져옴
        metrics = super().get_batch_loss_metrics(model, batch, train_eval)
        
        # 기본 손실 함수(loss)는 튜플/딕셔너리 형태로 반환되거나 내부 변수에 저장됨.
        # trl 버전에 따라 구현이 조금 다를 수 있으나 핵심 로직은 다음과 같습니다.
        policy_chosen_logps = metrics.get(f"{train_eval}/policy_chosen_logps", 0)
        policy_rejected_logps = metrics.get(f"{train_eval}/policy_rejected_logps", 0)
        reference_chosen_logps = metrics.get(f"{train_eval}/reference_chosen_logps", 0)
        reference_rejected_logps = metrics.get(f"{train_eval}/reference_rejected_logps", 0)

        # 2. Log Ratio 계산
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps
        logits = pi_logratios - ref_logratios
        
        # 3. 기본 DPO Loss 산출
        base_losses = -F.logsigmoid(self.beta * logits)
        
        # 4. 🔥 DA-DPO 핵심: Difficulty Weight 적용 🔥
        # 데이터 로더에서 전달된 가중치 텐서를 가져와 loss에 곱함
        weights = batch.get("difficulty_weight", torch.ones_like(base_losses))
        da_dpo_losses = base_losses * weights
        
        # 최종 Loss 평균 계산
        loss = da_dpo_losses.mean()
        
        # metrics 업데이트
        metrics[f"{train_eval}/loss"] = loss.item()
        return loss, metrics

# ==========================================
# 메인 학습 실행 함수
# ==========================================
def main():
    print(f"🔥 HC-DA-DPO 파인튜닝(Q-LoRA)을 시작합니다. 목표 모델: {MODEL_ID}")

    # 1. 4-bit 양자화 설정 (VRAM 절약의 핵심)
    # 32B 모델을 그대로 올리면 VRAM이 터지므로, NF4 포맷으로 압축하여 로드합니다.
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    print("모델과 프로세서를 로드하는 중...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",  # A100 8대에 자동 분산
        trust_remote_code=True
    )
    
    # Reference 모델은 학습시키지 않으므로, 동일한 4-bit 모델을 복사해서 메모리 효율화
    ref_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

    # 2. 모델을 양자화 학습에 맞게 준비 (Gradient Checkpointing 등)
    model = prepare_model_for_kbit_training(model)

    # 3. LoRA 어댑터 설정 (학습할 가중치 부위 지정)
    # VLM 특성상 Attention 모듈 전체(q, k, v, o)를 타겟으로 잡는 것이 성능이 좋습니다.
    peft_config = LoraConfig(
        r=64,
        lora_alpha=128,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 4. 데이터셋 로드
    dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
    
    # 5. Training Arguments (A100 8대 환경 최적화)
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,   # VRAM에 맞춰 조절
        gradient_accumulation_steps=8,   # Batch Size 뻥튀기
        learning_rate=5e-6,              # DPO는 LR을 매우 낮게 잡아야 함
        lr_scheduler_type="cosine",
        num_train_epochs=2,
        save_strategy="epoch",
        logging_steps=10,
        optim="paged_adamw_32bit",
        bf16=True,                       # A100이므로 bfloat16 필수
        remove_unused_columns=False,     # 중요! difficulty_weight 컬럼 삭제 방지
        gradient_checkpointing=True,
        report_to="tensorboard"
    )

    # 6. 트레이너 초기화 및 학습 시작
    trainer = DifficultyAwareDPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        beta=0.1,  # KL divergence 패널티 강도 (논문 실험 시 0.1 ~ 0.5 사이 조절)
        train_dataset=dataset,
        tokenizer=processor, # trl의 최근 버전은 processor를 직접 넘기기도 합니다.
        peft_config=peft_config,
    )

    print("🚀 본격적인 DA-DPO 학습 루프 진입!")
    trainer.train()
    
    # 7. 학습 완료 후 어댑터(LoRA Weight) 저장
    trainer.model.save_pretrained(os.path.join(OUTPUT_DIR, "final_adapter"))
    processor.save_pretrained(os.path.join(OUTPUT_DIR, "final_adapter"))
    print("🎉 학습 완료! LoRA 어댑터가 output 폴더에 저장되었습니다.")

if __name__ == "__main__":
    main()