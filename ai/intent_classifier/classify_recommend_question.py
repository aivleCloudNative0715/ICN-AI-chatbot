import pandas as pd
from tqdm import tqdm

from ai.shared.predict_intent_and_slots import predict_intent

# ✅ 파일 경로
FILE_PATH = "data/recommend_question_data.csv"
SAVE_PATH = "data/recommend_question_with_intent.csv"

# 📄 CSV/엑셀 파일 읽기
df = pd.read_csv(FILE_PATH)

# 🔁 각 질문에 대해 인텐트 예측
predicted_intents = []
predicted_probs = []

for question in tqdm(df['recommend_question'], desc="🔍 의도 예측 중"):
    intent, prob = predict_intent(question)
    predicted_intents.append(intent)
    predicted_probs.append(prob)

# 📎 결과 저장
df["predicted_intent"] = predicted_intents
df["intent_prob"] = predicted_probs
df.to_csv(SAVE_PATH, index=False)
print(f"✅ 예측 완료: {SAVE_PATH} 에 저장됨")