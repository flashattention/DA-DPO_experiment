# 🚀 HC-DA-DPO: Human-Calibrated Difficulty-Aware Direct Preference Optimization for Vision-Language Models

본 프로젝트는 초거대 시각-언어 모델(Vision-Language Model, VLM)의 문화적 환각(Cultural Hallucination) 현상을 억제하고, 인간 평가자의 가치관에 모델을 정밀하게 정렬(Alignment)하기 위한 **HC-DA-DPO** 파이프라인의 공식 구현체입니다.

단순히 정답/오답의 선호도만 학습하는 기존 DPO의 한계를 극복하기 위해, 모델의 **내재적 확신도(Intrinsic Confidence)**와 **인간의 체감 오차율(Human Calibration Penalty)**을 교차 검증하여 샘플별 **난이도 가중치(Difficulty Weight, $\omega$)**를 동적으로 산출합니다.

## 🧠 1. Theoretical Framework (수학적 배경)

### 1.1. Intrinsic Logprob Margin ($M$)
Reference Model이 정답($y_w$)과 오답($y_l$)에 대해 느끼는 내재적 확신도의 차이입니다. 텍스트 길이가 다를 수 있으므로 토큰 수($|y|$)로 나누어 정규화합니다.
$$M = \frac{1}{|y_w|} \sum_{t=1}^{|y_w|} \log \pi_{\text{ref}}(y_w^t | x, y_w^{<t}) - \frac{1}{|y_l|} \sum_{t=1}^{|y_l|} \log \pi_{\text{ref}}(y_l^t | x, y_l^{<t})$$
* $M > 0$: 모델이 이미 정답을 확신하는 쉬운 문제.
* $M < 0$: 모델이 오답을 더 그럴싸하게 여기는 어려운(환각) 문제.

### 1.2. Human Calibration Penalty ($H$)
인간 평가자와 기존 모델 간의 다차원적 의견 불일치 정도를 수치화한 페널티 항입니다.
$$H = 1.0 - \text{overall\_match\_rate}$$

이때 `overall_match_rate`는 서로 다른 척도를 가진 8개의 평가 항목을 0.0~1.0 사이의 부분 일치도(Partial Score)로 정규화하여 산술 평균한 종합 지표입니다. 단순 이진 판별을 넘어, 오차 거리에 비례한 연속적 패널티를 부여합니다.
* **범주형 (Yes/No):** 일치 시 $1.0$, 불일치 시 $0.0$ (항목: c, d, g)
* **정량적 척도 (1~5점):** $1.0 - \frac{|\text{diff}|}{4.0}$ (항목: a1, a2, e_cultural_specificity)
* **난이도 척도 (1~3점):** $1.0 - \frac{|\text{diff}|}{2.0}$ (항목: e_human_difficulty)
* **문자열 유사도:** `difflib.SequenceMatcher` 기반 텍스트 유사도 $0.0 \sim 1.0$ (항목: a3)

### 1.3. Difficulty Weight ($\omega$)
모델이 헷갈려 할수록($M$ 감소), 인간과 의견이 다를수록($H$ 증가) 가중치가 커지도록 시그모이드 함수를 적용합니다. ($\alpha$, $\beta$는 하이퍼파라미터)
$$\omega = \sigma(-\alpha M + \beta H)$$

### 1.4. HC-DA-DPO Objective Function
최종적으로 산출된 가중치 $\omega$를 기존 DPO 손실 함수에 결합하여, 모델이 '어렵고 혼동하기 쉬운' 맥락에 학습 역량을 집중하도록 유도합니다.
$$L_{\text{DA-DPO}} = -\omega \cdot \log \sigma (\beta \cdot (M_{\text{policy}} - M_{\text{ref}}))$$

---

## 💻 2. System Requirements (시스템 요구사항)
* **OS:** Ubuntu 20.04 / 22.04
* **GPU:** 8 x NVIDIA A100 (80GB) - *Total 640GB VRAM*
* **CUDA:** 11.8 or higher
* **Key Libraries:** `vllm`, `transformers`, `trl`, `peft`, `deepspeed`, `accelerate`

---

## 🏗️ 3. Pipeline Overview (파이프라인 구조)

본 프로젝트는 총 5개의 독립된 스크립트로 구성되며, 데이터 생성부터 논문 시각화까지 End-to-End로 동작합니다.

### 🔹 Script 1: Data Generation & Split (`1_generate_dpo_pairs.py`)
* **핵심 역할:** 압도적인 지능을 가진 초거대 교사 모델(235B)을 이용해, 모델 학습에 쓸 고품질의 '정답(Chosen) 및 오답(Rejected) 추론 과정(`<think>`)'을 만들어내는 데이터 증강 단계입니다.
* **작동 원리:**
  * 준비된 인간 평가 데이터(`merged_eval_data.jsonl`)를 로드합니다.
  * 235B vLLM 서버에 비동기(Async) API를 호출하여, 인간이 내린 평가를 바탕으로 "왜 이것이 문화적으로 맞고/틀린지" 논리적으로 설명하는 텍스트를 대필받습니다.
  * **[데이터 누수 방지]** 동일한 이미지가 Train과 Test에 동시에 들어가지 않도록 **Image ID 기준으로 90%의 학습 데이터와 10%의 평가 데이터로 엄격하게 분리(Split)**합니다.
* **출력물:** `train_dpo_dataset.json`, `test_dpo_dataset.json`

### 🔹 Script 2: Difficulty Weight Calculation (`2_calc_logprob.py`)
* **핵심 역할:** 1번에서 만든 데이터를 타겟 모델(32B)에게 보여주고 내재적 난이도를 수학적으로 채점하는 단계입니다.
* **작동 원리:**
  * 32B 베이스 모델을 Hugging Face로 로드하여 공식 ChatML 템플릿을 적용합니다.
  * 정답과 오답 텍스트에 대한 Token-level Logprob을 추출하여 **내재적 확신도 오차($M$)**를 계산합니다.
  * 인간 체감 오차율($H$, Overall Match Rate)과 $M$을 결합하여, 이 문제의 최종 **난이도 가중치($\omega$)**를 0~1 사이 값으로 산출합니다.
* **출력물:** `train_dpo_with_weights.json` (가중치 $\omega$가 포함된 최종 학습 데이터)

### 🔹 Script 3: Q-LoRA Fine-Tuning (`3_qlora_fine_tuning.py`)
* **핵심 역할:** 산출된 난이도 가중치($\omega$)를 적용하여, 32B 모델의 가중치를 실제로 업데이트하는 커스텀 분산 학습 단계입니다.
* **작동 원리:**
  * 4-bit 양자화된 32B 모델을 8대의 A100 GPU에 분산 적재(`DeepSpeed`/`Accelerate`)합니다.
  * `trl` 라이브러리의 표준 `DPOTrainer`를 오버라이딩합니다.
  * 난이도 가중치($\omega$)가 높은(어려운) 문제일수록 Loss(벌점)에 직접 곱해지도록 하여, **모델이 혼동하기 쉬운 문화적 맥락에 학습 역량을 집중**하게 만듭니다.
* **출력물:** `output_da_dpo_model/` (새롭게 학습된 LoRA 어댑터 가중치)

### 🔹 Script 4: Evaluation & Inference (`4_evaluate_and_benchmark.py`)
* **핵심 역할:** 개조된 32B 모델이 얼마나 똑똑해졌는지, 격리해두었던 10%의 Test Set으로 검증 및 벤치마크 평가를 수행하는 단계입니다.
* **작동 원리:**
  * 학습된 LoRA 어댑터를 32B 베이스 모델에 씌웁니다.
  * 10%의 Test Set 이미지를 입력하여 새로운 답변을 직접 생성하게 합니다.
  * 생성된 답변을 바탕으로 Baseline 모델과 비교하여 얼마나 정답에 가깝게 맞춰졌는지(Alignment) 객관적인 수치로 채점합니다.
* **출력물:** `evaluation_results.csv` (문항별 예측 결과 및 채점 데이터)

### 🔹 Script 5: Visualization for Publication (`5_visualize_metrics.py`)
* **핵심 역할:** 평가 결과를 분석하여, 제안 방법론(HC-DA-DPO)의 우수성을 입증하는 논문 게재용 고해상도 그래프를 자동 생성하는 단계입니다.
* **작동 원리:**
  * GPU 없이 Pandas와 Matplotlib를 사용하여 CSV 결과 파일을 분석합니다.
  * 난이도(Difficulty) 구간별 성능 방어율, 오차 밀도 함수 등을 계산합니다.
  * "문제가 어려워질수록 Baseline은 성능이 급락하지만, 제안 모델은 강건하게 방어한다"는 핵심 주장을 시각화합니다.
* **출력물:** `figures/` 폴더 내 고해상도(300 DPI) `.pdf` 그래프 파일들

---

## 🚀 4. Execution Guide (실행 가이드)

무거운 딥러닝 파이프라인의 특성상 `nohup`을 이용해 백그라운드에서 순차적으로 실행해야 합니다. 출력 버퍼링을 방지하기 위해 `-u` 옵션을 사용합니다.

### Step 0: 로그 디렉토리 생성
```bash
mkdir -p logs
mkdir -p output_da_dpo_model
mkdir -p figures
```

### Step 1: DPO 데이터 증강 (vLLM 구동 필수)
*주의: 이 단계에서는 `localhost:8000`에 Qwen-235B vLLM 서버가 띄워져 있어야 합니다.*
```bash
nohup python -u src/1_generate_dpo_pairs.py > logs/1_generate.log 2>&1 &

# 실시간 모니터링
tail -f logs/1_generate.log
```

### Step 2: 난이도(Logprob) 가중치 연산
**⚠️ 매우 중요:** 이 스크립트는 `transformers`를 사용하여 32B 모델을 직접 VRAM에 올리므로, **반드시 Step 1에서 사용한 vLLM 서버 프로세스를 종료(Kill)하여 VRAM을 비운 뒤 실행**해야 합니다.
```bash
nohup python -u src/2_calc_logprob.py > logs/2_calc_logprob.log 2>&1 &
```

### Step 3: HC-DA-DPO 분산 학습 (Q-LoRA)
8대의 A100 GPU 리소스를 100% 활용하기 위해 `accelerate`를 사용합니다.
```bash
# 최초 1회 환경 설정
accelerate config default

# 8 Multi-GPU 파인튜닝 실행
PYTHONUNBUFFERED=1 nohup accelerate launch --num_processes 8 src/3_qlora_fine_tuning.py > logs/3_qlora.log 2>&1 &

# GPU 사용량 모니터링
watch -n 1 nvidia-smi
```

### Step 4: 모델 추론 및 벤치마크 평가
```bash
nohup python -u src/4_evaluate_and_benchmark.py > logs/4_evaluate.log 2>&1 &
```

### Step 5: 논문용 메트릭 시각화
```bash
nohup python -u src/5_visualize_metrics.py > logs/5_visualize.log 2>&1 &
```
*완료 후 `figures/` 폴더에 생성된 `.pdf` 파일들을 논문에 삽입하십시오.*

---

## 📈 5. Expected Results (기대 효과)
본 파이프라인을 성공적으로 적용하면, 다음과 같은 결과를 얻을 수 있습니다.
* 쉬운 문제(Low Difficulty)에서는 Baseline 모델과 유사한 높은 성능(Alignment) 유지.
* **어려운 문제(High Difficulty)에서는 Baseline의 성능이 급락하는 반면, HC-DA-DPO 모델은 강건한(Robust) 성능 방어율을 보여줌으로써 본 제안 방법론의 우수성을 수학적·실증적으로 증명함.**

---
*Created for Top-tier NLP/CV Conference Submission*