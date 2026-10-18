# HC-DA-DPO: Human-Calibrated Difficulty-Aware DPO for a VLM Evaluator

한국 문화 VQA 문항을 채점하는 VLM 평가 모델(`Qwen/Qwen3-VL-32B-Thinking`)을 **사람 검수자의 채점 기준에 맞추기 위한** DPO 계열 정렬 실험 코드입니다. 표준 DPO, 난이도 인지 DPO(DA-DPO), 그리고 사람-모델 채점 일치율로 보정한 HC 변형들을 **동일한 학습 조건**에서 비교하는 Ablation을 수행합니다.

> 이 문서는 레포의 실제 코드(`src/`, `run_step*.sh`) 동작을 기준으로 작성했습니다. 실험 결과 수치는 레포에 포함되어 있지 않습니다.

---

## 1. 문제 설정

- **과제**: 이미지와 VQA 문항(키워드, 질문, 보기, 정답)을 입력받아, 평가 모델이 `<think>` 추론과 항목별 채점 JSON을 출력합니다.
- **문제**: 베이스 모델의 채점이 사람 검수자의 채점과 어긋납니다(특히 문화 특수성, 체감 난이도).
- **목표**: 사람의 채점과 그 근거를 선호 응답(chosen)으로, 모델 자신의 채점을 비선호 응답(rejected)으로 두고 DPO로 정렬합니다.

### 채점 항목과 일치율 (`src/0_generate_baseline_validations.py`)

| 항목 | 형식 | 일치 점수 |
|---|---|---|
| `a1_keyword_cultural` | 1–5 | 1 − \|차이\|/4 |
| `a2_image_keyword_alignment` | 1–5 | 1 − \|차이\|/4 |
| `a3_keyword_english` (키워드 영문 번역) | 텍스트 | 문자열 유사도(SequenceMatcher) |
| `c_question_image_relevance` | yes/no | 일치 1, 불일치 0 |
| `d_answer_accuracy` | correct/incorrect/ambiguous | 일치 1, 한쪽만 ambiguous 0.5, 반대 0 |
| `e_cultural_specificity` | 1–5 | 1 − \|차이\|/4 |
| `e_human_difficulty` | easy/medium/hard | 1 − \|차이\|/2 |
| `g_text_required` (OCR 필요 여부) | yes/no | 일치 1, 불일치 0 |

`overall_match_rate` = 8개 일치 점수의 평균. 키워드 번역은 별도로 서빙 중인 모델에 사람/모델 번역을 1–5점으로 비교하게 해 참고 지표로 남깁니다.

---

## 2. 파이프라인

| 단계 | 스크립트 | 내용 |
|---|---|---|
| 0 | `src/0_generate_baseline_validations.py` | 베이스 32B 모델(vLLM 서빙)로 전체 문항 채점, 사람 채점과의 항목별 일치율 계산 |
| 1 | `src/1_generate_dpo_pairs.py` | 이미지 단위 9:1 train/test 분리, 선호 쌍 생성 |
| 2 | `src/2_calc_dynamic_metrics.py` | 참조 모델 로그확률로 난이도 계산, 7가지 샘플 가중치 산출 |
| 3 | `src/3_train.py` | 가중치 열별로 QLoRA DPO 학습 (7개 설정) |
| 4 | `src/4_compare_adapters.py` | BF16 베이스 + 학습된 어댑터로 테스트셋 추론, 사람 채점과의 일치율 계산 |
| 5 | `src/5_run_evaluation.py` | LLM 심판의 pairwise swap test로 베이스라인 대비 승/무/패 집계 |

### 2.1 선호 쌍 생성 (Step 1)

- **분리**: 데이터 누수를 막기 위해 문항이 아니라 **이미지 기준**으로 train 90% / test 10%를 나눕니다(`random_state=42`).
- **제외**: 모델 채점이 사람과 완전히 일치한 문항(`overall_match_rate == 1.0`)은 학습할 신호가 없으므로 제외합니다.
- **prompt**: 이미지 + 키워드·질문·보기·정답 + "적절성을 평가하시오"
- **chosen**: `Qwen3-VL-235B-A22B-Thinking`이 **사람 채점 결과가 나온 이유를 역추적해 작성한 `<think>`** + 사람 채점 JSON
- **rejected**: 베이스 32B 모델이 실제로 생성한 `<think>` + 모델 채점 JSON

### 2.2 샘플 가중치 (Step 2)

참조 모델(32B, BF16)로 chosen과 rejected의 로그확률을 계산합니다. Thinking 모델의 긴 응답을 고려해 길이 보정을 적용합니다.

```
logp(y) = Σ log p(token) / len(y)^0.8
m       = logp(chosen) − logp(rejected)     # 참조 모델 기준 마진
mr      = overall_match_rate                # 사람-모델 채점 일치율
hp      = 1 − mr
ds      = σ(−m)                              # 난이도 점수 (마진이 작을수록 어려움)
α = 1.0, β = 0.8
```

| 설정 | 가중치 열 | 수식 | 의도 |
|---|---|---|---|
| DPO | `weight_dpo` | 1 | 균등 가중치 |
| DA-DPO | `weight_da_dpo` | σ(−α·m) | 참조 모델이 이미 chosen을 선호하는 쉬운 쌍은 낮게, 어려운 쌍은 높게 |
| HC-DPO | `weight_hc_dpo` | σ(β·(mr − 0.5)) | 사람-모델 일치율만으로 고정 보정 |
| HC-DA-DPO | `weight_hc_da_dpo` | σ(−α·m − β·hp) | 난이도 + 사람 불일치 보정의 단순 결합 |
| + Opt1 | `weight_hc_da_dpo_opt1` | σ(−α·m − β·hp·ds) | 사람 보정의 강도를 난이도에 비례시켜, 쉬운 쌍에서는 보정이 거의 사라지게 함 |
| + Opt3 | `weight_hc_da_dpo_opt3` | σ(−α·m + β·mr) | 일치율 항을 더하는 변형 |
| + Opt1 + Opt3 | `weight_hc_da_dpo_opt1_3` | σ(−α·m − β·hp·ds + β·mr) | 두 변형의 결합 |

원래 DA-DPO 논문([arXiv:2601.00623](https://arxiv.org/abs/2601.00623))은 쌍별 난이도로 DPO의 보상 스케일(β)을 조절하지만, 이 구현은 난이도를 **샘플별 손실 가중치**로 반영합니다.

### 2.3 학습 (Step 3)

`WeightedDPOTrainer`(TRL `DPOTrainer` 확장)가 샘플별 DPO 손실에 가중치를 곱합니다. 가중치는 데이터셋 전체 평균으로 정규화해 평균 1이 되도록 맞춥니다. 7개 설정 모두 아래 조건이 동일합니다.

| 항목 | 값 |
|---|---|
| 베이스 모델 | `Qwen/Qwen3-VL-32B-Thinking` |
| 양자화 | 4-bit NF4 QLoRA (double quant, compute dtype BF16) |
| LoRA | r=16, alpha=32, dropout=0.05 |
| LoRA 대상 | 언어 모델과 비전 인코더의 `q/k/v/o/gate/up/down_proj` |
| DPO β | 0.8 |
| 학습률 | 5e-7, cosine, warmup 10% |
| 에폭 | 1 |
| 배치 | GPU당 1 × gradient accumulation 16 × 4 GPU = 실효 64 |
| 최대 길이 | 4096 |
| 옵티마이저 | `paged_adamw_32bit` |
| 기타 | BF16 mixed precision, gradient checkpointing, FlashAttention 2 |
| 이미지 해상도 | min 256·28·28, max 1024·28·28 pixels |
| 분산 | `torchrun --nproc_per_node=4` (DDP) |
| 로깅 | Weights & Biases |

### 2.4 평가 (Step 4–5)

- **Step 4**: BF16 베이스 모델에 각 어댑터를 붙여 테스트셋을 Simple/Detailed 두 가지 프롬프트로 추론하고, 사람 채점과의 `overall_match_rate`를 기록합니다. 현재 평가 대상은 `Baseline`, `Standard_DPO`, `DA_DPO`, `HC_DA_DPO`입니다(Opt 변형은 `MODELS_TO_EVALUATE`에 추가해야 평가됩니다).
- **Step 5**: LLM 심판(`gpt-4o` 또는 `gpt-4o-mini`)이 베이스라인과 대상 모델의 Detailed JSON 응답을 사람 정답과 함께 비교합니다. A/B 순서를 바꿔 두 번 묻고, **두 번 모두 같은 쪽을 고른 경우에만 승리**로 집계해 위치 편향을 줄입니다. 결과는 승/무/패 비율 그래프로 저장됩니다.

---

## 3. 실행 방법

### 환경 변수

```env
# 프로젝트 루트의 .env
WANDB_API_KEY="your_wandb_api_key_here"
OPENAI_API_KEY="your_openai_api_key_here"
```

### 순서

```bash
# Step 0: 베이스 32B 모델 서빙 (vLLM, BF16, tensor parallel 4)
bash vllm-qwen3-vl-32b-thinking.sh
python src/0_generate_baseline_validations.py --save_thoughts

# Step 1: 235B Thinking 모델 서빙 (tensor parallel 8) 후 선호 쌍 생성
bash vllm-qwen3-vl-235b-a22b-thinking.sh
python src/1_generate_dpo_pairs.py

# Step 2: 샘플 가중치 계산 (백그라운드 실행)
bash run_step2_weights.sh

# Step 3: 7개 설정 순차 학습 → runs/exp_<timestamp>/ 에 어댑터 저장
bash run_step3_train.sh

# Step 4: GPU별 분할 추론 (rank마다 하나씩 실행)
for r in 0 1 2 3; do
  python src/4_compare_adapters.py --exp_dir runs/exp_<timestamp> --rank $r --world_size 4 &
done; wait

# Step 5: 분할 결과 병합 + pairwise swap test
bash run_step5_evaluation.sh runs/exp_<timestamp>      # 심판: gpt-4o
```

`run_step4_compare_adapters.sh`는 이름과 달리 Step 5와 같은 병합·평가 스크립트를 `gpt-4o-mini` 심판으로 실행합니다.

### vLLM 서빙 설정

| 모델 | tensor parallel | max_model_len | 비고 |
|---|---|---|---|
| Qwen3-VL-32B-Thinking | 4 | 32768 | `gpu_memory_utilization 0.90`, 프롬프트당 이미지 1장 |
| Qwen3-VL-235B-A22B-Thinking | 8 | 8192 | BF16 |

---

## 4. 구현 노트

- **샘플 가중치 전달**: TRL의 기본 콜레이터는 커스텀 스칼라 열을 처리하지 못하므로, `DPOWeightCollator`가 `sample_weight`를 따로 묶고 Trainer가 손실 계산 직전에 꺼내 쓴 뒤 바로 비웁니다.
- **dtype 정합성**: 혼합 정밀도에서 가중치 텐서를 손실 텐서의 dtype에 맞춘 뒤 곱합니다.
- **가변 이미지 패치**: Qwen-VL은 이미지마다 패치 수가 달라 `pixel_values`와 `image_grid_thw`를 `stack`이 아니라 `cat`으로 합칩니다.
- **EOS 보존**: `padding_side="right"`, `truncation_side="left"`로 설정해 길이를 넘을 때 프롬프트 앞부분을 자르고 응답 끝의 EOS를 보존합니다.
- **길이 보정 로그확률**: Thinking 모델은 응답 길이 편차가 커서, 합 로그확률을 `len^0.8`로 나눠 길이 편향을 줄입니다.

## 5. 한계

- 레포에 학습 결과와 평가 수치가 포함되어 있지 않습니다.
- chosen의 `<think>`는 사람이 직접 쓴 것이 아니라, 사람 채점을 근거로 235B 모델이 작성한 추론입니다.
- 학습은 4-bit 양자화 베이스에서, 평가는 BF16 베이스에서 수행되어 학습과 추론의 정밀도가 다릅니다.
- 가중치 하이퍼파라미터(α, β, 길이 보정 지수)는 고정값이며 별도 탐색 결과는 포함되어 있지 않습니다.
