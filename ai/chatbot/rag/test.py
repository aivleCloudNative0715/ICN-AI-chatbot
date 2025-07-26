from sentence_transformers import SentenceTransformer

model = SentenceTransformer("dragonkue/snowflake-arctic-embed-l-v2.0-ko")
dummy_text = "안녕하세요, 테스트입니다."
embedding = model.encode(dummy_text).tolist()

print(f"모델이 생성하는 임베딩 차원: {len(embedding)}")