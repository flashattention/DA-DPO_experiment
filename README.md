# 🚀 HC-DA-DPO: Human-Calibrated Difficulty-Aware Direct Preference Optimization for Vision-Language Models

본 프로젝트는 초거대 시각-언어 모델(Vision-Language Model, VLM)의 **문화적 환각(Cultural Hallucination)** 현상을 억제하고,  
인간 평가자의 정밀한 가치관에 모델을 정렬(Alignment)하기 위한 **HC-DA-DPO 파이프라인의 공식 구현체**입니다.

기존 **DPO (Direct Preference Optimization)** 가 데이터의 난이도와 상관없이 균등한 가중치를 부여함으로써 발생하는 문제를 해결하기 위해,  
모델의 **내재적 확신도 (Intrinsic Confidence)** 와 **인간의 체감 오차율 (Human Calibration Penalty)** 을 결합한 **동적 난이도 가중치( \( \omega \) )** 산출 기법을 제안합니다.

---

## 🧠 1. Theoretical Framework & Rationale

### 1.1. The Problem: Central Clustering & Confidence Bias

- **Central Clustering:** 프롬프트 엔지니어링 기반의 SFT 모델이 문화적 특이성(Cultural Specificity) 질문에 대해 깊이 있는 답변을 피하고 '보통(3점)' 수준의 중립적인 답변으로 회피하는 현상.  
- **Confidence Bias:** 모델이 문화적으로 틀린 정보를 생성하면서도 높은 Logprob(확신)을 보이는 '근거 없는 자신감' 상태의 환각 문제.

---

### 1.2. Metric 1: Intrinsic Logprob Margin ( \( M \) )

Reference Model이 정답( \( y_w \) )과 오답( \( y_l \) )에 대해 산출한 토큰당 평균 로그 확률의 차이입니다.

\[
M = \frac{1}{|y_w|} \sum_{t=1}^{|y_w|} \log \pi_{\text{ref}}(y_w^t | x, y_w^{<t}) 
- \frac{1}{|y_l|} \sum_{t=1}^{|y_l|} \log \pi_{\text{ref}}(y_l^t | x, y_l^{<t})
\]

\( M \) 값이 작을수록 모델은 두 응답을 구분하기 어려워하며, 이는 모델 관점의 난이도를 의미합니다.

---

### 1.3. Metric 2: Human Calibration Penalty ( \( H \) )

인간 검수자가 부여한 8개 차원의 평가 데이터를 통합하여 모델의 실제 정렬 오차를 산출합니다.

\[
H = 1.0 - \text{overall\_match\_rate}
\]

---

### 1.4. Proposed Objective: HC-DA-DPO

모델이 헷갈려 하거나( \( M \downarrow \) ), 인간의 기준에서 크게 벗어난( \( H \uparrow \) ) 샘플에 대해 학습 강도를 동적으로 조절합니다.

\[
\omega = \sigma(-\alpha M + \beta H)
\]

최종 손실 함수는 다음과 같이 정의됩니다.

\[
L_{\text{HC-DA-DPO}}(\theta; \pi_{\text{ref}}) = 
-\mathbb{E}_{(x, y_w, y_l) \sim D} 
\left[ 
  \omega \cdot \log \sigma 
  \left( 
    \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} 
    - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} 
  \right) 
\right]
\]

---

## 🏗️ 2. Ablation Study & Experimental Design

본 연구는 단계적 비교 실험을 통해 각 요소의 기여도를 정량적으로 분석합니다.

| 모델 구분 | 가중치 로직 ( \( \omega \) ) | 실험 목적 |
|------------|-----------------------------|------------|
| **Baseline** | N/A | 파인튜닝 전 기초 성능 측정 |
| **Standard DPO** | \( \omega = 1.0 \) | 일반 선호도 학습 시의 한계 확인 |
| **DA-DPO** | \( \omega = \sigma(-\alpha M) \) | 모델 내재적 난이도 반영 효과 검증 |
| **HC-DA-DPO** | \( \omega = \sigma(-\alpha M + \beta H) \) | 인간 가치관 결합을 통한 환각 교정 효과 검증 |

---

## 🛠️ 3. Implementation Details

### 3.1. On-Policy CoT Distillation (235B Teacher)

- **Impedance Mismatch 해결:** GPT-4o 대신 동일 계열의 `Qwen3-VL-235B-A22B-Thinking` 모델을 활용하여 가상의 추론 과정(`<think>`)을 생성, 지식 전이 효율 극대화.

### 3.2. Architecture & MLOps Pipeline

- **VLM Architecture Support:**  
  멀티모달 네이티브(`AutoModelForVision2Seq`) 클래스를 적용하여 Vision 인코더와 LLM 디코더 간의 완전한 호환성 확보.  
- **Automated Background Execution:**  
  `nohup` 기반의 자체 백그라운드 실행 쉘 스크립트 구축으로 SSH 단절 시에도 안정적인 학습 보장.  
- **Environment Management:**  
  `.env` 파일을 통한 API Key(`WANDB_API_KEY`, `OPENAI_API_KEY`) 자동 주입 시스템 적용.

---

## 🚀 4. Execution Guide (Automated Pipeline)

### ⚙️ Prerequisites (환경 변수 설정)

프로젝트 루트 경로에 `.env` 파일을 생성하고 다음 키를 입력합니다.

```bash
WANDB_API_KEY="your_wandb_api_key"
OPENAI_API_KEY="your_openai_api_key"
```

---

### Step 1: Teacher Distillation & Split

```bash
python src/1_generate_dpo_pairs.py
```

- **On-Policy Reasoning Generation:**  
  235B 초거대 멀티모달 모델을 Teacher로 활용하여, 주어진 시각 정보 기반의 심층 추론(Chain-of-Thought) 데이터를 생성합니다.  
- **Preference Pair Construction:**  
  인간 검수 데이터와 Teacher 모델의 추론 논리를 결합하여 Chosen( \( y_w \) ) 및 Rejected( \( y_l \) ) 쌍을 구축합니다.  
- **Image-Centric Splitting:**  
  동일한 Image ID가 Train/Test에 중복되지 않도록 Strict Split 수행.

---

### Step 2: Weight Calculation (Logprob Calibration)

```bash
python src/2_calc_logprob.py
```

- **Baseline Self-Diagnostic:**  
  학습 대상 모델이 스스로 Chosen/Rejected 응답의 난이도( \( M \) )를 정량화.  
- **Metric Integration:**  
  내재 확률값과 Calibration Penalty( \( H \) )를 결합하여 샘플별 **가중치( \( \omega \) )** 계산.  
- **Efficiency:**  
  사전 연산 후 `.jsonl` 메타데이터로 캐싱하여 학습 중 부하 경감.

---

### Step 3: Model Training (Ablation Stage)

```bash
bash run_step3_train.sh
```

- **Background Orchestration:**  
  8 × A100 GPU 환경에서 Standard, DA, HC-DA 세 가지 모델을 순차 학습.  
- **Optimization Tech:**  
  DeepSpeed Stage 2 + 4-bit NF4 Q-LoRA, Zero-Redundancy Optimizer로 효율 극대화.  
- **VLM Native Loading:**  
  `AutoModelForVision2Seq` 기반 Vision–Language 결합 구조로 로드.

---

### Step 4: 8-way Evaluation (Inference Stage)

```bash
bash run_step4_evaluate.sh [EXP_FOLDER]
```

- **Multi-Ablation Inference:**  
  학습 완료된 체크포인트별 응답 자동 생성.  
- **Zero-Shot Cultural VQA:**  
  외부 지식 개입 없이 시각 정보와 내재 문화 지식만으로 추론.

---

### Step 5: GPT-4o Grading & Visualization (Final Stage)

```bash
bash run_step5_visualize.sh [EXP_FOLDER]
```

- **LLM-as-a-Judge:**  
  GPT-4o를 인간 심판으로 활용하여 각 모델의 정답률 및 문화적 환각 억제 수준 평가.  
- **Radar & Loss Curve Visualization:**  
  난이도별 성능 향상 폭과 Alignment Score, Win-rate 시각화.

---

## 📈 5. Expected Results

- **Alignment Target:**  
  고난도 문화적 질문 영역에서 Baseline 대비 유의미한 Alignment Score 향상.  
- **Hallucination Mitigation:**  
  Calibration Penalty를 통한 Confidence Bias 억제.  
- **Robustness Visualization:**  
  난이도 기반 성능 곡선을 통해 HC-DA-DPO의 강건성 증명.

---

**Created for Top-tier NLP/CV Conference Submission (CVPR, NeurIPS, ICCV, ACL)**
