# 🎓 CEFR Level Classifier: Synthetic Data Distillation from Qwen3

> **Synthetic Data Distillation을 통해 구축된 CEFR (Common European Framework of Reference for Languages) 텍스트 분류기입니다.**

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-dksysd%2Fcefr--classifier-blue)](https://huggingface.co/dksysd/cefr-classifier)
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

## 📖 개요 (Overview)

이 프로젝트는 **CEFR 레벨 분류기** (A1 ~ C2)를 학습시키기 위한 파이프라인을 구현합니다. 거대 언어 모델(LLM)의 언어적 능력을 더 작고 효율적인 인코더 모델로 전이시키는 **Synthetic Data Distillation(합성 데이터 증류)** 방식을 채택했습니다.

1.  **Teacher (데이터 생성):** `Qwen3-32B` 모델이 CEFR 가이드라인에 기반하여 언어적으로 다양한 고품질 합성 데이터를 생성합니다.
2.  **Student (분류기):** `DeBERTa-v3-large` 모델이 이 증류된 데이터셋으로 미세 조정(Fine-tuning)되어 경량화된 분류기로서 작동합니다.

또한, 언어 능숙도 레벨의 점진적인 특성(예: A1은 C2보다 A2에 더 가깝음)을 반영하기 위해 **Ordinal Label Smoothing Loss**를 모델 아키텍처에 통합했습니다.

## ✨ 주요 기능 (Key Features)

*   **🧬 합성 데이터 증류 (Synthetic Data Distillation):** **Qwen3**를 사용하여 병렬 코퍼스(Parallel Corpora)와 지시 튜닝(Instruction) 데이터셋을 생성하는 자동화된 파이프라인입니다. 이 방법은 교사 모델의 지식을 데이터셋으로 효과적으로 전이하여 주제의 다양성을 확보하고 노이즈를 줄입니다.
*   **📉 서열 레이블 스무딩 (Ordinal Label Smoothing, OLS):** CEFR 레벨 간의 거리에 기반하여 예측 오차를 계산하는 사용자 정의 손실 함수로, 능숙도 간의 서열 관계를 학습에 반영합니다.
*   **🔍 의미론적 중복 제거 (Semantic Deduplication):** `SentenceTransformers`를 사용하여 생성 과정에서 의미적으로 중복되는 샘플을 필터링하여 데이터셋의 다양성을 높입니다.
*   **🛡️ 안정적인 설계:** 재시도 로직, 스트리밍 저장 등 안정성을 고려한 설계와 최신 도구(`uv` 패키지 매니저)로 구축되었습니다.

## 📂 프로젝트 구조

```text
cefr-classifier/
├── data/                   # 생성된 데이터셋 (JSONL)
│   ├── instructions/       # 지시 튜닝용 QA 쌍
│   └── parallel_corpus/    # 레벨별 텍스트 재작성 태스크
├── generation/             # 데이터 생성 파이프라인
│   └── generator_pipeline.py  # Qwen3 기반 생성 스크립트
├── training/               # 모델 학습
│   ├── loss.py             # 사용자 정의 Ordinal Label Smoothing Loss
│   └── train.py            # Grid Search 기능이 포함된 학습 스크립트
├── .gitignore
├── pyproject.toml          # 의존성 관리 (uv 사용)
└── README.md
```

## 🚀 설치 (Installation)

이 프로젝트는 빠른 패키지 관리를 위해 **[uv](https://github.com/astral-sh/uv)**를 사용합니다.

```bash
# 리포지토리 클론
git clone https://github.com/YOUR_USERNAME/cefr-classifier.git
cd cefr-classifier

# 의존성 설치
uv sync
# 또는 pip 사용 시 (requirements.txt가 있는 경우)
# pip install -r requirements.txt
```

*참고: 효율적인 데이터 생성을 위해 GPU 지원이 활성화된 `llama-cpp-python`이 필요합니다.*

## ⚡ 빠른 시작 (추론)

`transformers` 라이브러리를 사용하여 Hugging Face에서 학습된 모델을 직접 사용할 수 있습니다.

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 모델 및 토크나이저 로드
model_id = "dksysd/cefr-classifier"  # 실제 모델 ID로 변경 필요
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id)

# 레이블 매핑
id2label = {0: "A1", 1: "A2", 2: "B1", 3: "B2", 4: "C1", 5: "C2"}

def predict_cefr(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        logits = model(**inputs).logits
    
    predicted_class_id = logits.argmax().item()
    return id2label[predicted_class_id]

# 테스트
text = "The cat sat on the mat."
print(f"Level: {predict_cefr(text)}") # 예상 결과: A1
```

## 🛠️ 사용법 (Usage)

### 1. 데이터 생성 (Distillation)

로컬 LLM(Qwen3)을 사용하여 합성 학습 데이터를 생성합니다. 이 단계는 LLM의 지식을 구조화된 데이터셋으로 증류합니다.

```bash
cd generation
python generator_pipeline.py \
    --samples 100 \
    --model_repo "unsloth/Qwen3-32B-GGUF" \
    --output_dir "../data"
```

### 2. 모델 학습

생성된 데이터를 사용하여 분류기를 학습시킵니다. 스크립트는 자동 Grid Search를 지원합니다.

```bash
cd training
python train.py \
    --instruction_file "../data/instructions/train.jsonl" \
    --parallel_file "../data/parallel_corpus/train.jsonl" \
    --output_dir "./results" \
    --lr 2e-5
```

## 🔬 실험 및 인사이트 (Experiment & Insights)

모델의 성능을 검증하기 위해 하이퍼파라미터(LoRA Rank, Alpha, Learning Rate)에 대한 광범위한 **Grid Search**를 수행했습니다.

> **💡 핵심 발견: 데이터 품질을 통한 견고성 (Robustness)**
> 실험 결과, 모델은 다양한 하이퍼파라미터 설정(LoRA Rank, Learning Rate)에서도 일관된 성능을 유지했습니다.
> 이러한 **높은 견고성**은 성능의 주요 요인이 모델의 용량이나 특정 튜닝 파라미터보다는 **Qwen3가 생성한 합성 데이터의 품질과 일관성**에 있음을 시사합니다. 증류 과정이 명확한 언어적 패턴을 성공적으로 전이시켰기 때문에, 학생 모델(DeBERTa)이 하이퍼파라미터 변화에 덜 민감하게 반응한 것으로 분석됩니다.

## 🧠 기술적 세부사항 (Technical Details)

### 합성 데이터 증류 (Synthetic Data Distillation)
Soft-target(logits)으로부터 학습하는 전통적인 지식 증류(Knowledge Distillation)와 달리, 이 프로젝트는 **응답 기반 증류(Response-based Distillation)**를 사용합니다. "Teacher" 모델(Qwen3)은 자신의 추론 및 언어적 지식이 담긴 정제된 데이터셋을 생성합니다. "Student" 모델(DeBERTa)은 이 합성 데이터셋을 학습함으로써 추론 시 Teacher 모델의 많은 리소스를 요구하지 않고도 해당 지식을 효과적으로 학습하여 높은 성능을 달성합니다.

### 서열 레이블 스무딩 손실 (Ordinal Label Smoothing Loss)
표준 Cross-Entropy는 모든 오답 클래스를 동등하게 취급합니다. 하지만 **A1을 C2로 예측하는 것**은 **A1을 A2로 예측하는 것**보다 더 큰 오차로 간주되어야 합니다.
우리의 사용자 정의 손실 함수는 레이블 확률을 인접한 클래스에 분배하여 이를 반영합니다:

```python
# 타겟 분포의 개념적 시각화
Target: B1 (Index 2)
Standard One-Hot: [0,    0,    1,    0,    0, 0]
Ordinal Smooth:   [0, 0.05,  0.9, 0.05,    0, 0]
```

## 📜 라이선스 (License)

이 프로젝트는 **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)** 라이선스 하에 배포됩니다.
비상업적 목적으로 자료를 공유하고 각색할 수 있으며, 적절한 출처를 명시하고 변경된 내용을 동일한 라이선스로 배포해야 합니다.