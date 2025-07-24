import pandas as pd
import re

def clean_text(text):
    """
    KoBERT 기반 전처리에 적합하도록 특수문자 제거 및 공백 정리
    """
    # 한글, 영문, 숫자, 공백만 남기기
    text = re.sub(r"[^\uAC00-\uD7A3a-zA-Z0-9\s]", "", str(text))
    # 다중 공백 제거
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def preprocess_intent_csv(input_path: str, output_path: str):
    # CSV 로드
    df = pd.read_csv(input_path)

    # 필수 컬럼 확인 및 NaN 제거
    df.dropna(subset=['intent', 'question'], inplace=True)

    # 문자열로 변환 후 전처리
    df['question'] = df['question'].astype(str).apply(clean_text)

    # 중복 제거
    df.drop_duplicates(subset=['intent', 'question'], inplace=True)

    # 저장
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"✅ 전처리 완료: {output_path}")

# 실행 예시
if __name__ == "__main__":
    preprocess_intent_csv("intent_dataset.csv", "intent_dataset_cleaned.csv")

