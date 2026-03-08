import os
import torch
import argparse
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, 
    AutoProcessor, 
    BitsAndBytesConfig, 
    TrainingArguments
)
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import DPOTrainer

# ==========================================
# [경로 설정]
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# 🌟 2번 스크립트가 뱉어낸 '최신(latest)' 데이터를 바라보도록 수정
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "train_hc_dpo_pairs_latest.jsonl")

MODEL_ID = "Qwen/Qwen3-VL-32B-Thinking"

# ==========================================
# 🌟 커스텀 DPO 트레이너
# ==========================================
class WeightedDPOTrainer(DPOTrainer):
    def get_batch_loss_metrics(self, model, batch, train_eval="train"):
        loss, chosen_rewards, rejected_rewards = super().get_batch_loss_metrics(model, batch, train_eval)
        
        if "sample_weight" in batch:
            weights = torch.tensor(batch["sample_weight"], device=loss.device, dtype=loss.dtype)
            loss = loss * weights
            
        return loss, chosen_rewards, rejected_rewards

def main():
    # 1. 터미널 인자 파싱 (쉘 스크립트에서 값을 넘겨줌)
    parser = argparse.ArgumentParser(description="Ablation Training Script")
    parser.add_argument("--weight_col", type=str, required=True, help="학습에 사용할 가중치 컬럼명")
    parser.add_argument("--output_dir", type=str, required=True, help="결과물이 저장될 폴더명")
    parser.add_argument("--run_name", type=str, required=True, help="W&B 기록용 실험 이름") # 🌟 W&B용 추가
    args = parser.parse_args()

    # 🌟 쉘 스크립트에서 넘겨준 타임스탬프 폴더 경로를 그대로 사용
    output_path = os.path.join(PROJECT_ROOT, args.output_dir)
    print(f"\n🚀 [학습 시작] 가중치: {args.weight_col} -> 저장 폴더: {output_path}")

    # 2. 데이터 로드 및 가중치 매핑
    dataset = load_dataset("json", data_files=DATA_PATH, split="train")
    dataset = dataset.map(lambda x: {"sample_weight": x[args.weight_col]})

    # 3. 4-bit 양자화 및 모델 로드
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=bnb_config, device_map="auto", trust_remote_code=True
    )
    model = prepare_model_for_kbit_training(model)

    # 4. LoRA 설정
    peft_config = LoraConfig(
        r=64,
        lora_alpha=128,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)

    # 5. TrainingArguments
    training_args = TrainingArguments(
        output_dir=output_path,
        per_device_train_batch_size=2,   
        gradient_accumulation_steps=8,
        optim="paged_adamw_32bit",
        learning_rate=5e-6,
        lr_scheduler_type="cosine",
        num_train_epochs=2,
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
        gradient_checkpointing=True,     
        remove_unused_columns=False,
        report_to="wandb",                  # 🌟 W&B 로깅 켜기
        run_name=args.run_name              # 🌟 W&B에 표시될 실험 이름
    )

    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

    # 6. 트레이너 실행
    trainer = WeightedDPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=processor.tokenizer,
        peft_config=peft_config,
        beta=0.1, 
        max_prompt_length=1024,
        max_length=2048
    )

    trainer.train()
    
    # 7. 저장
    trainer.save_model()
    processor.save_pretrained(output_path)
    print(f"✅ {args.weight_col} 학습 완료! 어댑터가 저장되었습니다.\n")

if __name__ == "__main__":
    main()