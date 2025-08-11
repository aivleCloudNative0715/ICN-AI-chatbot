print("스크립트 시작!")  # 코드 최상단에 추가
from pymongo import MongoClient
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from tqdm import tqdm  # 진행률 표시

# .env 파일에서 환경변수 불러오기
load_dotenv()
mongo_uri = os.getenv("MONGO_URI")

# 1. 임베딩 모델 로드
print("임베딩 모델을 로드 중입니다... (최초 실행 시 다운로드로 인해 시간이 소요될 수 있습니다.)")
try:
    embedding_model = SentenceTransformer('dragonkue/snowflake-arctic-embed-l-v2.0-ko')
    print("임베딩 모델 로드 완료.")
except Exception as e:
    print(f"임베딩 모델 로드 중 오류 발생: {e}")
    exit()

# 2. MongoDB 연결
client = None
try:
    client = MongoClient(mongo_uri)
    db = client["AirBot"]

    # 기존 컬렉션
    original_collection_name = "AirlineVector"
    airport_collection = db["Airline"]
    airport_vectors_collection = db[original_collection_name]

    # 임시 컬렉션
    temp_collection_name = original_collection_name + "_tmp"
    airport_vectors_temp_collection = db[temp_collection_name]

    print(f"원본 컬렉션: {airport_collection.name}")
    print(f"기존 벡터 컬렉션: {airport_vectors_collection.name}")
    print(f"임시 벡터 컬렉션: {airport_vectors_temp_collection.name}")

    # 임시 컬렉션 초기화
    airport_vectors_temp_collection.drop()
    print(f"임시 컬렉션 '{temp_collection_name}' 초기화 완료.")

    # 데이터 불러오기
    documents_cursor = airport_collection.find({})
    total_documents = airport_collection.count_documents({})

    processed_documents_count = 0
    documents_to_insert = []

    for doc in tqdm(documents_cursor, total=total_documents, desc="Processing Airlines"):
        try:
            airline_code = doc.get('airline_code', '')
            airline_name_kor = doc.get('airline_name_kor', '')
            airline_contact = doc.get('airline_contact', '')

            text_to_embed = f"항공사 코드 {airline_code}는 {airline_name_kor}입니다. 전화번호는 {airline_contact}입니다"

            if not text_to_embed.strip():
                print(f"경고: _id {doc.get('_id')} 문서에서 임베딩할 텍스트가 없습니다.")
                continue

            embedding = embedding_model.encode(text_to_embed).tolist()

            new_doc = {
                "original_id": doc['_id'],
                "text_content": text_to_embed,
                "embedding": embedding
            }
            documents_to_insert.append(new_doc)
            processed_documents_count += 1

            if len(documents_to_insert) >= 1000:
                airport_vectors_temp_collection.insert_many(documents_to_insert)
                documents_to_insert = []

        except Exception as doc_error:
            print(f"\n문서 처리 중 오류 발생 (ID: {doc.get('_id')}): {doc_error}")
            continue

    if documents_to_insert:
        airport_vectors_temp_collection.insert_many(documents_to_insert)

    print(f"\n총 {processed_documents_count}개의 문서를 '{temp_collection_name}'에 저장 완료.")

    # 기존 컬렉션 삭제 & 스왑
    airport_vectors_collection.drop()
    airport_vectors_temp_collection.rename(original_collection_name)
    print(f"컬렉션 교체 완료: '{temp_collection_name}' → '{original_collection_name}'")

except Exception as e:
    print(f"MongoDB 작업 중 오류 발생: {e}")

finally:
    if client:
        client.close()
        print("\nMongoDB 연결이 종료되었습니다.")
