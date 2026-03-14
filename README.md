# 🚀 HC-DA-DPO: Human-Calibrated Difficulty-Aware Direct Preference Optimization for Vision-Language Models

[![Conference](https://img.shields.io/badge/Target-Top--tier_NLP%2FCV-red.svg)]()
[![Model](https://img.shields.io/badge/Base_Model-Qwen3--VL--32B--Thinking-blue.svg)]()
[![Python](https://img.shields.io/badge/Python-3.10+-green.svg)]()
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)]()

본 프로젝트는 초거대 시각-언어 모델(Vision-Language Model, VLM)의 **문화적 환각(Cultural Hallucination)** 현상을 억제하고, 인간 평가자의 정밀한 가치관에 모델을 정렬(Alignment)하기 위한 **HC-DA-DPO 파이프라인의 공식 구현체**입니다.

기존 **DPO (Direct Preference Optimization)** 가 데이터의 난이도와 상관없이 균등한 가중치를 부여함으로써 발생하는 문제를 해결하기 위해, 모델의 **내재적 확신도 (Intrinsic Confidence)** 와 **인간의 체감 오차율 (Human Calibration Penalty)** 을 결합한 **동적 난이도 가중치( $\omega$ )** 산출 기법을 제안합니다.

---

## 🧠 1. Theoretical Framework & Rationale

### 1.1. The Problem: Central Clustering & Confidence Bias

- **Central Clustering:** 프롬프트 엔지니어링 기반의 SFT 모델이 문화적 특이성(Cultural Specificity) 질문에 대해 깊이 있는 답변을 피하고 '보통(3점)' 수준의 중립적인 답변으로 회피하는 현상.  
- **Confidence Bias:** 모델이 문화적으로 틀린 정보를 생성하면서도 높은 Logprob(확신)을 보이는 '근거 없는 자신감' 상태의 환각 문제.

---

### 1.2. Metric 1: Intrinsic Logprob Margin ( $M$ )

Reference Model이 정답( $y_w$ )과 오답( $y_l$ )에 대해 산출한 토큰당 평균 로그 확률의 차이입니다.

$$M = \frac{1}{|y_w|} \sum_{t=1}^{|y_w|} \log \pi_{\text{ref}}(y_w^t | x, y_w^{<t}) - \frac{1}{|y_l|} \sum_{t=1}^{|y_l|} \log \pi_{\text{ref}}(y_l^t | x, y_l^{<t})$$

$M$ 값이 작을수록 모델은 두 응답을 구분하기 어려워하며, 이는 모델 관점의 난이도를 의미합니다.

---

### 1.3. Metric 2: Human Calibration Penalty ( $H$ )

인간 검수자가 부여한 8개 차원의 평가 데이터를 통합하여, 모델의 실제 답변이 인간의 문화적 기준과 얼마나 어긋나는지 정렬 오차를 산출합니다.

$$H = 1.0 - \text{overall\_match\_rate}$$

---

### 1.4. Proposed Objective: HC-DA-DPO

모델이 헷갈려 하거나( $M \downarrow$ ), 인간의 기준에서 크게 벗어난( $H \uparrow$ ) 샘플에 대해 학습 강도를 동적으로 조절합니다.

$$\omega = \sigma(-\alpha M + \beta H)$$

최종 손실 함수는 다음과 같이 정의됩니다.

$$L_{\text{HC-DA-DPO}}(\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l) \sim D} \left[ \omega \cdot \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right) \right]$$

---

## 🏗️ 2. Ablation Study & Experimental Design

본 연구는 단계적 비교 실험을 통해 각 요소의 기여도를 정량적으로 분석합니다.

| 모델 구분 | 가중치 로직 ( $\omega$ ) | 실험 목적 |
|------------|-----------------------------|------------|
| **Baseline** | N/A | 파인튜닝 전 기초 성능 측정 |
| **Standard DPO** | $\omega = 1.0$ | 일반 선호도 학습 시의 한계 확인 |
| **DA-DPO** | $\omega = \sigma(-\alpha M)$ | 모델 내재적 난이도 반영 효과 검증 |
| **HC-DA-DPO** | $\omega = \sigma(-\alpha M + \beta H)$ | 인간 가치관 결합을 통한 환각 교정 효과 검증 |

---

## 🛠️ 3. Implementation Details

### 3.1. On-Policy CoT Distillation 

- **Impedance Mismatch 해결:** 타사 API(GPT-4o 등) 대신 동일 아키텍처의 초대형 파라미터 모델(`Qwen3-VL-235B-A22B-Thinking`)을 활용하여 가상의 추론 과정(`<think>`)을 생성, 지식 전이 효율을 극대화합니다.

### 3.2. Architecture & MLOps Pipeline

- **High-Throughput Generation:** `vLLM` 서버 구동 쉘 스크립트와 `AsyncOpenAI` 비동기 파이프라인을 결합하여 평가 속도 극대화.
- **VLM Architecture Support:** 멀티모달 네이티브(`AutoModelForImageTextToText`) 클래스를 적용하여 호환성 확보.  
- **Multi-GPU Distributed Inference:** 평가 병목 해소를 위해 데이터셋을 청크 단위로 자동 분할하고 N개의 GPU에 텐서를 분산 로드하여 초고속 평가 수행.

---

## 🚀 4. Execution Guide (Automated Pipeline)

### ⚙️ Prerequisites (환경 변수 설정)

프로젝트 루트 경로에 `.env` 파일을 생성하고 다음 키를 입력합니다.

```env
WANDB_API_KEY="your_wandb_api_key"
OPENAI_API_KEY="your_openai_api_key"
```

---

### Step 0: Baseline Diagnostic & Human Calibration Data Generation

Baseline 모델(32B)을 vLLM으로 서빙하고, 비동기 파이프라인을 통해 인간 검수 데이터와의 오차를 계산합니다.

```bash
# 1. 터미널 A: 32B 모델 vLLM 서버 구동
bash vllm-qwen3-vl-32b-thinking.sh

# 2. 터미널 B: Baseline 평가 및 리포트 자동 생성 파이프라인 실행
python src/0_generate_baseline_validations.py --save_thoughts
```

- **Hard Negative Extraction:** `e_cultural_specificity` 오차가 3점 이상인 취약 샘플을 자동 추출하며, 완료 즉시 `generate_baseline_report.py`가 연쇄 실행되어 리포트를 구성합니다.

---

### Step 1: Teacher Distillation & Split

초대형 Teacher 모델(235B)을 활용해 DPO 학습을 위한 선호도 데이터 쌍(Preference Pairs)을 생성합니다.

```bash
# 1. 터미널 A: 235B 모델 vLLM 서버 구동
bash vllm-qwen3-vl-235b-a22b-thinking.sh

# 2. 터미널 B: DPO Pair 생성 파이프라인 실행
python src/1_generate_dpo_pairs.py
```

- **Preference Pair Construction:** 인간 검수 데이터(Chosen)와 Teacher 모델의 오답 추론(Rejected)을 결합.  
- **Image-Centric Splitting:** Train/Test 셋 간 이미지 중복 유출(Data Leakage)을 차단.

---

### Step 2: Weight Calculation (Logprob Calibration)

```bash
python src/2_calc_logprob.py
```

- **Metric Integration:** 모델의 내재 확률값($M$)과 Step 0에서 도출한 Calibration Penalty($H$)를 융합하여 각 데이터 쌍에 적용할 **최종 난이도 가중치($\omega$)**를 사전 계산하고 캐싱합니다.

---

### Step 3: Model Training (Ablation Stage)

```bash
bash run_step3_train.sh
```

- **Background Orchestration:** 다중 GPU 환경에서 Standard, DA, HC-DA 모델(LoRA 어댑터)을 순차적으로 자동 학습합니다.  
- **Optimization Tech:** DeepSpeed Stage 2 + 4-bit NF4 Q-LoRA가 적용되어 있습니다.

---

### Step 4: Multi-GPU Parallel Evaluation (Inference Stage)

```bash
bash run_step4_evaluate.sh [EXP_FOLDER]
```

- **Distributed GPU Processing:** 학습이 완료된 모델들을 순회하며, 여러 대의 GPU(`rank 0~N`)를 활용해 테스트 데이터셋을 병렬 추론합니다.  
- **Zero-Shot Cultural VQA:** 외부 지식(RAG) 개입 없이 시각 정보와 모델의 내재적 지식만으로 Simple / Detailed 모드 평가 수행.

---

### 🌟 Step 5: Top-Tier Optimized Evaluation & Visualization

본 연구는 평가의 학술적 엄밀성을 위해 기존의 절대평가(Pointwise) 방식을 배제하고, 리뷰어들의 비판을 원천 차단하는 **고도화된 상대평가 파이프라인**을 구동합니다.

```bash
python src/5_run_evaluation.py --exp_dir runs/exp_YOUR_EXP_ID --judge_model gpt-4o --num_workers 10
```

- **Avoidance of Self-Preference Bias (핵심):** Qwen 계열 모델이 스스로를 높게 평가하는 '자기 선호 편향'을 차단하기 위해, 이종 아키텍처인 `GPT-4o` (또는 로컬 `Llama-3.1-70B`)를 심판(Judge)으로 사용합니다.
- **Pairwise Swap Test:** LLM의 고질적인 위치 편향(A 응답을 선호하는 현상)을 상쇄하기 위해, A/B 응답 순서를 스왑하여 교차 평가(Cross-validation)하고 두 결과가 일치할 때만 승률(Win-Rate)로 인정합니다.
- **Parallel & Fault-Tolerant API:** `ThreadPoolExecutor`를 통한 병렬 처리로 평가 시간을 10배 이상 단축시켰으며, API Rate Limit 발생 시 지수 백오프(Exponential Backoff)와 실시간 체크포인팅(Checkpointing)을 통해 안정적인 평가를 보장합니다.
- **Output:** 논문 첨부용 고해상도 Stacked Bar Chart (`fig1_pairwise_win_rate.pdf`) 및 Loss Curve 자동 생성.

---

## 📈 5. Expected Results

- **Alignment Target:** 고난도 문화적 질문 영역에서 Baseline 대비 유의미한 Alignment Score 향상 (Win-Rate 50% 이상 달성).  
- **Hallucination Mitigation:** Calibration Penalty를 통한 Confidence Bias(근거 없는 확신) 현상 억제.  
- **Robustness Visualization:** 난이도 기반 성능 곡선을 통해 제안된 HC-DA-DPO의 강건성 입증.

---

**Created for Top-tier NLP/CV Conference Submission (CVPR, NeurIPS, ICCV, ACL)**