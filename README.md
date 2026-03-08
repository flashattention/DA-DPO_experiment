# 🚀 HC-DA-DPO: Human-Calibrated Difficulty-Aware Direct Preference Optimization for Vision-Language Models

본 프로젝트는 초거대 시각-언어 모델(Vision-Language Model, VLM)의 **문화적 환각(Cultural Hallucination)** 현상을 억제하고, 인간 평가자의 정밀한 가치관에 모델을 정렬(Alignment)하기 위한 **HC-DA-DPO** 파이프라인의 공식 구현체입니다.

기존 DPO(Direct Preference Optimization)가 데이터의 난이도와 상관없이 균등한 가중치를 부여함으로써 발생하는 '쉬운 샘플에 대한 오버피팅'과 '환각 샘플에 대한 정렬 실패' 문제를 해결하기 위해, 모델의 **내재적 확신도(Intrinsic Confidence)**와 **인간의 체감 오차율(Human Calibration Penalty)**을 결합한 **동적 난이도 가중치($\omega$)** 산출 기법을 제안합니다.

---

## 🧠 1. Theoretical Framework & Rationale

### 1.1. The Problem: Central Clustering & Confidence Bias
* **Central Clustering:** 프롬프트 엔지니어링 기반의 SFT 모델이 문화적 특이성(Cultural Specificity) 질문에 대해 깊이 있는 답변을 피하고 '보통(3점)' 수준의 중립적인 답변으로 회피하는 현상.
* **Confidence Bias:** 모델이 문화적으로 틀린 정보를 생성하면서도 높은 Logprob(확신)을 보이는 '근거 없는 자신감' 상태의 환각 문제.

### 1.2. Metric 1: Intrinsic Logprob Margin ($M$)
모델이 스스로 느끼는 변별력을 측정합니다. Reference Model이 정답($y_w$)과 오답($y_l$)에 대해 산출한 토큰당 평균 로그 확률의 차이입니다.
$$M = \frac{1}{|y_w|} \sum_{t=1}^{|y_w|} \log \pi_{\text{ref}}(y_w^t | x, y_w^{<t}) - \frac{1}{|y_l|} \sum_{t=1}^{|y_l|} \log \pi_{\text{ref}}(y_l^t | x, y_l^{<t})$$
* $M$ 값이 작을수록 모델은 두 응답을 구분하기 어려워하며, 이는 **모델 관점의 난이도**를 의미합니다.

### 1.3. Metric 2: Human Calibration Penalty ($H$)
인간 검수자가 부여한 8개 차원의 정성/정량 평가 데이터를 통합하여 모델의 실제 정렬 오차를 산출합니다.
$$H = 1.0 - \text{overall\_match\_rate}$$
`overall_match_rate`는 다음의 지표들을 포함합니다:
* **Categorical Alignment:** 문화적 관련성, 시각적 환각 여부, 정답 일치도 (Yes/No)
* **Quantitative Scalability:** 정렬도, 상세도, 문화적 특이성 (1~5 Scale)
* **Reasoning Integrity:** 추론 논리의 인간-모델 유사도 (String Similarity & Logic Match)

### 1.4. Proposed Objective: HC-DA-DPO
모델이 헷갈려 하거나($M \downarrow$), 인간의 기준에서 크게 벗어난($H \uparrow$) 샘플에 대해 학습 강도를 시그모이드 형태로 동적 조절합니다.
$$\omega = \sigma(-\alpha M + \beta H)$$
최종 손실 함수는 다음과 같이 정의됩니다:
$$L_{\text{HC-DA-DPO}}(\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l) \sim D} \left[ \omega \cdot \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right) \right]$$



---

## 🏗️ 2. Ablation Study & Experimental Design

본 연구는 단계적 Ablation Study를 통해 각 요소의 기여도를 정량적으로 증명합니다.

| 모델 구분 | 가중치 로직 ($\omega$) | 핵심 학습 기여 |
| :--- | :--- | :--- |
| **Baseline** | N/A | SFT(Supervised Fine-Tuning) 상태의 기초 성능 |
| **Standard DPO** | $\omega = 1.0$ (Constant) | 일반적인 선호도 정렬 (Easy-sample Overfitting 존재) |
| **DA-DPO** | $\omega = \sigma(-\alpha M)$ | 모델이 모호함을 느끼는 경계 샘플에 대한 변별력 강화 |
| **HC-DA-DPO** | **$\omega = \sigma(-\alpha M + \beta H)$** | **인간 가치관 기반의 환각 교정 및 문화적 정밀 정렬** |

---

## 🛠️ 3. Implementation Details (Pipeline)

### 3.1. On-Policy CoT Distillation (235B Teacher)
* **Impedance Mismatch 해결:** GPT-4o와 같은 타 가문의 모델 대신, 동일한 언어적 DNA(Tokenizer/Architecture)를 공유하는 **Qwen3-VL-235B-A22B-Thinking** 모델을 활용하여 가상의 추론 과정(`<think>`)을 생성.
* **Style Alignment:** 하위 모델(32B)이 상위 모델의 고도화된 논리 구조를 '자신의 말투'로 자연스럽게 습득하게 함.

### 3.2. Data Engineering
* **Sample Size:** 정제된 7,900건의 고품질 인간 검수 데이터.
* **Strict Split:** Image ID 기준 **90%(Train) : 10%(Test)** 분할. 테스트 데이터는 학습 중 절대 노출되지 않으며 최종 평가(Step 5)에서만 사용됨.

### 3.3. Distributed Training Infrastructure
* **Hardware:** 8 x NVIDIA A100 (80GB), NVLink 활성화.
* **Optimization Stack:** * `DeepSpeed Stage 2`를 통한 Zero Redundancy Optimizer 적용.
    * `Flash-Attention 2` 및 `4-bit NF4` 양자화 기반 Q-LoRA 학습.
    * `WeightedDPOTrainer`: 샘플별 $\omega$를 배치의 Loss에 Element-wise로 결합하는 커스텀 엔진.

---

## 🚀 4. Execution Guide

### Step 1: Teacher Distillation & Split
```bash
# 235B vLLM을 통해 인간 평가 기반의 Ground Truth 추론 로직 생성
python src/1_generate_dpo_pairs.py
```

### Step 2: Weight Calculation (Logprob Calibration)
```bash
# 32B 모델을 통해 M(Intrinsic)과 H(Human)를 교차 연산하여 Weight 산출
# VRAM 확보를 위해 이전 단계의 vLLM 서버 종료 필수
python src/2_calc_logprob.py
```

### Step 3: End-to-End Ablation (Master Script)
```bash
# 학습(3종) -> 추론 -> GPT-4o 채점 -> 논문용 시각화를 일괄 수행
chmod +x run_ablation.sh
./run_ablation.sh
```

---

## 📈 5. Expected Scholarly Impact
* **Robustness:** 난이도가 높은(High $H$) 문화적 질문에서 Baseline 대비 **15% 이상의 Alignment Score** 향상 입증.
* **Hallucination Mitigation:** 인간의 Calibration 페널티가 모델의 '근거 없는 자신감'을 억제하고 정답 확률 분포를 개선함을 시각적으로 증명.
* **Scalability:** 타 언어 및 문화권으로의 자동화된 VLM 정렬 파이프라인 확장 가능성 제시.

---
*This repository is intended for submission to top-tier conferences (CVPR, NeurIPS, ICCV, ACL).*