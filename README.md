# 3rd_new_drug

**Jump AI(py) 2025 : 제3회 AI 신약개발 경진대회**  
**2nd Place / 502 Teams – ChemBERTa pIC50 Prediction**

---

ChemBERTa 기반의 **ASK1 (MAP3K5) IC50 → pIC50 회귀 모델** 학습 및 추론 프로젝트입니다.  
학습 코드와 추론 코드를 분리하고, **YAML 설정 파일 기반**으로 실험을 관리할 수 있도록 구성되어 있습니다.

---

## 🎯 Project Goals

- **SMILES 입력 기반 pIC50 회귀**
- **Scaffold 기반 Stratified Group K-Fold**
- **HuggingFace Trainer 기반 안정적인 학습**
- **Fold별 best checkpoint를 활용한 소프트 앙상블 추론**

---

## 📁 Project Structure

```text
chemberta_project/
├── src/                    # 핵심 로직 (import용)
│   ├── __init__.py
│   ├── model.py            # 모델 로딩
│   ├── dataset.py          # 데이터 전처리 / split
│   ├── trainer.py          # 학습 루프 (CV)
│   ├── losses.py           # (확장용) custom loss
│   └── utils.py            # 공용 함수
│
├── train.py                # 학습 실행 스크립트
├── inference.py            # 추론 / 제출 파일 생성
│
├── configs/                # 설정 파일 (코드 수정 없이 실험 제어)
│   ├── train.yaml
│   └── submit.yaml
│
├── assets/                 # 모델 / 토크나이저 (보통 gitignore)
│   ├── model.pt
│   └── tokenizer/
│
├── requirements.txt        # 실행 환경 고정
├── .gitignore
├── .gitattributes
└── README.md
```

---

## 🛠 Environment Setup

Python 3.9+ 권장

```
pip install -r requirements.txt
```

---

## 📊 Dataset Format
ID, Smiles 2columns

```
ID, Smiles
TEST_000, CCO..
TEST_001, CCN..
```

---

## 🚀 Training
### 1️⃣ 설정 파일 수정

configs/train.yaml에서 다음 항목을 조절할 수 있습니다.

- 모델 이름 (예: DeepChem/ChemBERTa-77M-MTR)
- Batch size / Epoch / Learning rate
- Fold 수
- 출력 디렉터리

### 2️⃣ 학습 실행

```
python train.py --config configs/train.yaml
```

학습 완료 후 생성되는 파일:

- Fold별 best checkpoint
- OOF prediction (oof_*.csv)
- manifest.json (추론 시 사용)

---

## 📦 Inference & Submission

```
python inference.py --config configs/submit.yaml
```

- Fold별 best checkpoint 로드
- pIC50 평균 → IC50(nM) 변환
- 소프트 앙상블 결과 저장

--- 

## 🧠 Model Details

- Backbone: ChemBERTa-77M-MTR
- Task: Regression (pIC50)
- Loss: MSE
- CV Strategy:
  - Murcko Scaffold 기반 Group
  - pIC50 bin 기반 Stratification

---

## 📌 Notes

- assets/, data/, ckpt/는 기본적으로 gitignore 대상
- 대용량 파일은 Git LFS 사용 권장
- losses.py는 custom loss 실험 시 확장 가능
