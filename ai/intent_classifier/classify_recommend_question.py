import pandas as pd
from tqdm import tqdm
import json

from ai.shared.predict_intent_and_slots import predict_intent, predict_with_bce
from shared.config import INTENT_CLASSIFICATION

# ✅ 파일 경로
FILE_PATH = "data/recommend_question_data.csv"
SAVE_PATH = "data/recommend_question_with_intent_filtered.csv"

# 📄 CSV/엑셀 파일 읽기
df = pd.read_csv(FILE_PATH)

# 🔁 각 질문에 대해 인텐트 예측
filtered_data = []

for idx, row in tqdm(df.iterrows(), total=len(df), desc="🔍 의도 예측 및 필터링 중"):
    question = row['recommend_question']
    original_intent_list = json.loads(row['intent_list'])
    
    result = predict_with_bce(question, threshold=INTENT_CLASSIFICATION["DEFAULT_THRESHOLD"], top_k_intents=3)
    
    # 복합 의도인지 확인
    is_multi_intent = result['is_multi_intent']
    
    # 단일 의도인 경우만 추가
    if not is_multi_intent:
        if result['all_top_intents']:
            predicted_intent, prob = result['all_top_intents'][0]
        else:
            predicted_intent, prob = "default", 0.0
        
        filtered_data.append({
            'intent_list': json.dumps([predicted_intent]),
            'recommend_question': question,
            'original_intent': json.dumps(original_intent_list),
            'predicted_intent': predicted_intent,
            'intent_prob': prob,
            'is_single_intent': True
        })
    else:
        # 복합 의도는 제외하지만 로그 출력
        high_confidence_intents = [intent for intent, _ in result['high_confidence_intents']]
        print(f"복합 의도 제외: '{question}' -> {high_confidence_intents}")

# 📎 결과를 DataFrame으로 변환 및 저장
filtered_df = pd.DataFrame(filtered_data)
filtered_df.to_csv(SAVE_PATH, index=False)

print(f"✅ 필터링 완료:")
print(f"   - 전체 질문: {len(df)}개")
print(f"   - 단일 의도 질문: {len(filtered_df)}개")
print(f"   - 복합 의도 제외: {len(df) - len(filtered_df)}개")
print(f"   - 저장 위치: {SAVE_PATH}")