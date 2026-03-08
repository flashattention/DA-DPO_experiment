# 🚀 HC-DA-DPO: Human-Calibrated Difficulty-Aware Direct Preference Optimization for Vision-Language Models

본 프로젝트는 초거대 시각-언어 모델(Vision-Language Model, VLM)의 문화적 환각(Cultural Hallucination) 현상을 억제하고, 인간 평가자의 가치관에 모델을 정밀하게 정렬(Alignment)하기 위한 **HC-DA-DPO** 파이프라인의 공식 구현체입니다.

단순히 정답/오답의 선호도만 학습하는 기존 DPO의 한계를 극복하기 위해, 모델의 **내재적 확신도(Intrinsic Confidence)**와 **인간의 체감 오차율(Human Calibration Penalty)**을 교차 검증하여 샘플별 **난이도 가중치(Difficulty Weight, $\omega$)**를 동적으로 산출하는 방법론을 제안합니다.

---

## 🧠 1. Theoretical Framework (수학적 배경 및 상세 지표)

### 1.1. Intrinsic Logprob Margin ($M$)
Reference Model이 정답($y_w$)과 오답($y_l$)에 대해 느끼는 내재적 확신도의 차이입니다. 텍스트 길이가 다를 수 있으므로 토큰 수($|y|$)로 나누어 정규화합니다.
$$M = \frac{1}{|y_w|} \sum_{t=1}^{|y_w|} \log \pi_{\text{ref}}(y_w^t | x, y_w^{<t}) - \frac{1}{|y_l|} \sum_{t=1}^{|y_l|} \log \pi_{\text{ref}}(y_l^t | x, y_l^{<t})$$

### 1.2. Human Calibration Penalty ($H$) 상세 계산
인간 평가자와 기존 모델 간의 다차원적 의견 불일치를 수치화합니다. $H = 1.0 - \text{overall\_match\_rate}$ 이며, `overall_match_rate`는 다음 8개 항목의 가중 평균입니다:

* **범주형 (Yes/No):** 일치 시 $1.0$, 불일치 시 $0.0$ (항목: c_cultural_relevance, d_visual_hallucination, g_final_answer)
* **정량적 척도 (1~5점):** $1.0 - \frac{|\text{diff}|}{4.0}$ (항목: a1_alignment, a2_detail, e_cultural_specificity)
* **난이도 척도 (1~3점):** $1.0 - \frac{|\text{diff}|}{2.0}$ (항목: e_human_difficulty)
* **문자열 유사도:** `difflib.SequenceMatcher` 기반 텍스트 유사도 $0.0 \sim 1.0$ (항목: a3_reasoning_logic)

### 1.3. Difficulty Weight ($\omega$) 및 손실 함수
$$\omega = \sigma(-\alpha M + \beta H)$$
최종 손실 함수는 가중치 $\omega$를 결합하여 모델이 '어렵고 인간과 동떨어진' 맥락에 학습 역량을 집중하게 합니다.
$$L_{\text{DA-DPO}} = -\omega \cdot \log \sigma (\beta \cdot (M_{\text{policy}} - M_{\text{ref}}))$$

---

## 🏗️ 2. Ablation Study Setup (비교 실험 설계)

본 프로젝트는 제안 방법론의 우수성을 증명하기 위해 4가지 실험 조건을 비교합니다:

| 모델 구분 | 설명 | 가중치($\omega$) 적용 방식 |
| :--- | :--- | :--- |
| **Baseline** | 파인튜닝되지 않은 32B SFT 모델 | N/A |
| **Standard DPO** | 일반적인 직접 선호도 최적화 | $\omega = 1.0$ (균등 가중치) |
| **DA-DPO** | 모델의 내재적 난이도만 고려 | $\omega = \sigma(-\alpha M)$ |
| **HC-DA-DPO** | **[제안]** 모델 난이도 + 인간 페널티 결합 | $\omega = \sigma(-\alpha M + \beta H)$ |

---

## 💻 3. System Requirements & Infrastructure

* **GPU:** 8 x NVIDIA A100 (80GB) - *Total 640GB VRAM*
* **Distributed Training:** `DeepSpeed Stage 2`, `Accelerate`
* **Optimization:** `4-bit NormalFloat` Quantization, `Paged Optimizer`
* **Inference:** `vLLM` (for 235B Teacher), `Hugging Face` (for 32B Target)

---

## 🛠️ 4. Pipeline Details (Script-by-Script)

### 🔹 Script 1: Data Generation & Split (`1_generate_dpo_pairs.py`)
* **235B Teacher 모델 활용:** 인간의 평가 로직을 기반으로 고품질 추론 과정(`<think>`)을 생성합니다.
* **Image-based Splitting:** 동일 이미지가 Train/Test에 섞이지 않도록 `Image ID` 기준 **90:10 분할**을 엄격히 적용하여 데이터 누수를 방지합니다.

### 🔹 Script 2: Weight Calculation (`2_calc_logprob.py`)
* **Logprob 추출:** 32B 베이스 모델을 로드하여 정답/오답의 토큰별 로그 확률을 계산합니다.
* **VRAM 관리:** `torch.cuda.empty_cache()` 및 `gc.collect()`를 매 배치마다 호출하여 32B 모델의 Logprob 연산 시 OOM을 방지합니다.

### 🔹 Script 3: Q-LoRA Fine-Tuning (`3_train.py`)
* **Weighted DPO:** `trl` 라이브러리의 `DPOTrainer`를 오버라이딩하여 샘플별 $\omega$를 Loss에 동적으로 곱해주는 커스텀 트레이너를 사용합니다.
* **W&B 연동:** 실험별 Loss 수렴 속도와 Reward Margin을 실시간 모니터링합니다.

### 🔹 Script 4: Batch Inference (`4_evaluate.py`)
* **Sequential Loading:** 4가지 Ablation 어댑터를 순차적으로 로드하여 추론을 수행합니다.
* **Robust Padding:** `PIL.Image` 로드 시 발생할 수 있는 `Incorrect padding` 에러를 방지하기 위해 정규화 전처리를 수행합니다.

### 🔹 Script 5: GPT-4o Scoring & Visualization (`5_visualize.py`)
* **LLM-as-a-Judge:** 생성된 텍스트를 GPT-4o API를 통해 0.0~5.0점으로 정교하게 채점합니다.
* **Journal-Ready Figures:** Matplotlib/Seaborn을 사용하여 300 DPI 고해상도 PDF 그래프를 생성합니다 (Robustness Curve, KDE Plot).

---

## 🚀 5. Execution Guide

### Step 0: 환경 설정
```bash
export OPENAI_API_KEY="sk-..."
export WANDB_API_KEY="..."
mkdir -p logs runs
```

### Step 1 & 2: 전처리 및 가중치 생성
```bash
# 235B vLLM 서버 구동 확인 후 실행
python src/1_generate_dpo_pairs.py
# vLLM 서버 종료 후 VRAM 확보 상태에서 실행
python src/2_calc_logprob.py
```

### Step 3: 마스터 파이프라인 가동 (학습-추론-채점-시각화)
본 쉘 스크립트는 3가지 Ablation 실험을 타임스탬프가 적용된 독립된 실험 폴더에 자동으로 보존하며 순차 실행합니다.

```bash
chmod +x run_ablation.sh
./run_ablation.sh
```

---

## 📈 6. Expected Results
* **Robustness in Hard Samples:** 인간이 '어렵다'고 느낀 문제(High $H$)에서 타 모델 대비 **15~20% 이상의 성능 방어율** 달성.
* **Cultural Alignment:** 문화적 맥락이 중요한 문항에서 환각(Hallucination) 발생 빈도의 유의미한 감소.
* **Scientific Proof:** 난이도별 성능 곡선에서 제안 모델(HC-DA-DPO)이 가장 강건한(Robust) 형태를 보임을 입증.

---
*Created for Top-tier NLP/CV Conference Submission (CVPR, NeurIPS, ACL)*