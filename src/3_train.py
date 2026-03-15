import os
import torch
import argparse
from datasets import load_dataset
from transformers import (
    AutoModelForImageTextToText, 
    AutoProcessor, 
    BitsAndBytesConfig
)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import DPOConfig, DPOTrainer

# ==========================================
# [경로 설정]
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "train_experiment_pairs_latest.jsonl")
MODEL_ID = "Qwen/Qwen3-VL-32B-Thinking"

def resolve_image_path(image_path: str) -> str:
    if os.path.isabs(image_path): return image_path
    for p in [os.path.join(PROJECT_ROOT, image_path), os.path.join(PROJECT_ROOT, "data", image_path)]:
        if os.path.exists(p): return p
    return ""

# ==========================================
# 🌟 [Fix 2] TRL 호환형 스마트 방어 Collator
# ==========================================
class DPOWeightCollator:
    def __init__(self, base_collator):
        self.base_collator = base_collator

    def __call__(self, features):
        weights = [f.pop("sample_weight") for f in features if "sample_weight" in f]
        
        # 🚨 TRL은 'prompt_input_ids', 'chosen_input_ids' 등을 생성함
        # 동적 키워드 탐색으로 문자열(String) 패닉을 원천 차단
        if any(any(k.endswith("input_ids") for k in f.keys()) for f in features):
            keys_to_remove = ["prompt", "chosen", "rejected", "image_path"]
            for f in features:
                for k in keys_to_remove:
                    f.pop(k, None)
        
        batch = self.base_collator(features)
        
        if weights:
            batch["sample_weight"] = torch.tensor(weights, dtype=torch.bfloat16)
        return batch

# ==========================================
# Custom DPOTrainer (수석 엔지니어 보강판)
# ==========================================
class WeightedDPOTrainer(DPOTrainer):
    def _prepare_inputs(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        prepared_inputs = super()._prepare_inputs(inputs)
        if "sample_weight" in prepared_inputs:
            # 원본 유지: 모델의 Forward 패스에 들어가지 않도록 pop 처리
            self.current_sample_weights = prepared_inputs.pop("sample_weight").to(self.model.device)
        else:
            self.current_sample_weights = None
        return prepared_inputs

    def dpo_loss(self, policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps):
        losses, chosen_rewards, rejected_rewards = super().dpo_loss(
            policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps
        )
        
        # Element-wise 가중치 곱 적용
        if getattr(self, 'current_sample_weights', None) is not None:
            # 🌟 [보강 1] Dtype 동기화: AMP 업캐스팅에 의한 정밀도 손실 및 충돌 방어
            safe_weights = self.current_sample_weights.to(dtype=losses.dtype)
            losses = losses * safe_weights
            
            # 🌟 [보강 2] State Bleeding 방어: 사용 완료된 가중치 즉시 소각
            self.current_sample_weights = None

        return losses, chosen_rewards, rejected_rewards

def format_for_multimodal_dpo(example, weight_col, global_mean_weight):
    file_uri = f"file://{example['resolved_path']}"
    
    prompt_msg = [{"role": "user", "content": [{"type": "image", "image": file_uri}, {"type": "text", "text": example["prompt"]}]}]
    chosen_msg = [{"role": "assistant", "content": [{"type": "text", "text": example["chosen"]}]}]
    rejected_msg = [{"role": "assistant", "content": [{"type": "text", "text": example["rejected"]}]}]
    
    normalized_weight = example[weight_col] / global_mean_weight

    return {
        "prompt": prompt_msg,
        "chosen": chosen_msg,
        "rejected": rejected_msg,
        "sample_weight": normalized_weight,
    }

def main():
    parser = argparse.ArgumentParser(description="Ablation Training Script for HC-DA-DPO")
    parser.add_argument("--weight_col", type=str, required=True, help="학습 가중치 컬럼")
    parser.add_argument("--output_dir", type=str, required=True, help="저장 디렉토리")
    parser.add_argument("--run_name", type=str, required=True, help="W&B 실험명")
    args = parser.parse_args()

    output_path = os.path.join(PROJECT_ROOT, args.output_dir)
    print(f"\n🚀 [학습 개시] 타겟 가중치: {args.weight_col} -> 저장소: {output_path}")

    # 1. 데이터 로드 및 1차 필터링 (결함 방어)
    raw_dataset = load_dataset("json", data_files=DATA_PATH, split="train")
    # Step 1: 디스크 I/O는 여기서 단 한 번만 수행하여 그 결과를 데이터셋에 캐싱(저장)합니다.
    raw_dataset = raw_dataset.map(
        lambda x: {"resolved_path": resolve_image_path(x["image_path"])},
        num_proc=4,
        desc="[I/O] Resolving image paths"
    )
    # Step 2: valid_raw_dataset의 화려한 부활.
    # 디스크 접근 없이 메모리 상의 문자열 비교만 하므로 순식간에 필터링됩니다.
    valid_raw_dataset = raw_dataset.filter(
        lambda x: x["resolved_path"] != "", 
        num_proc=4,
        desc="Filtering invalid data"
    )
    
    # Step 3: 불량 데이터가 100% 제거된 순백의 데이터셋에서 완벽한 가중치 평균을 계산합니다.
    weights_list = valid_raw_dataset[args.weight_col]
    global_mean_weight = sum(weights_list) / (len(weights_list) + 1e-8)
    
    # Step 4: 이미 검증된 경로와 완벽한 정규화 상수를 가지고 최종 DPO 포맷팅을 수행합니다.
    dataset = valid_raw_dataset.map(
        lambda x: format_for_multimodal_dpo(x, args.weight_col, global_mean_weight),
        remove_columns=valid_raw_dataset.column_names, 
        num_proc=4,
        desc="Formatting & Normalizing"
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    
    # 🚨 [Fix 3] DPO EOS 마스킹 붕괴 및 Runaway Generation 방어
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    # DPO 손실 함수가 길이를 맞출 때, 우측 패딩을 강제하여 시퀀스 끝의 실제 EOS가 마스킹되지 않게 보호합니다.
    processor.tokenizer.padding_side = "right" 
    
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, 
        quantization_config=bnb_config, 
        device_map={"": local_rank}, 
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16
    )

    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.use_cache = False
    
    model = prepare_model_for_kbit_training(model)
    model.enable_input_require_grads()

    peft_config = LoraConfig(
        r=64,
        lora_alpha=128,
        target_modules=r".*model\.layers.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)",
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    training_args = DPOConfig(
        output_dir=output_path,
        per_device_train_batch_size=1, 
        gradient_accumulation_steps=16,
        optim="paged_adamw_32bit",
        learning_rate=5e-6,
        lr_scheduler_type="cosine",
        num_train_epochs=2,
        logging_steps=5,
        save_strategy="epoch",
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}, 
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,   
        beta=0.1,
        max_length=4096,
        max_prompt_length=3072,
        dataloader_num_workers=4,           
        report_to="wandb",                  
        run_name=args.run_name              
    )

    trainer = WeightedDPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset, # 이미 필터링된 데이터셋 사용
        processing_class=processor,
        peft_config=peft_config,
    )
    
    trainer.data_collator = DPOWeightCollator(trainer.data_collator)

    trainer.train()
    trainer.save_model()
    processor.save_pretrained(output_path)
    print(f"✅ [{args.weight_col}] 수학적/연산적 무결성 훈련 완료 및 어댑터 저장 성공!\n")

if __name__ == "__main__":
    main()