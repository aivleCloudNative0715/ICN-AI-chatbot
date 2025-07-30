print("스크립트 시작!") # 코드 최상단에 추가
from pymongo import MongoClient
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from tqdm import tqdm # 진행률 표시를 위한 라이브러리

# .env 파일에서 환경변수 불러오기 (MONGO_URI)
load_dotenv()
mongo_uri = os.getenv("MONGO_URI")

# 1. 임베딩 모델 로드
print("임베딩 모델을 로드 중입니다... (최초 실행 시 다운로드로 인해 시간이 소요될 수 있습니다.)")
try:
    embedding_model = SentenceTransformer('dragonkue/snowflake-arctic-embed-l-v2.0-ko')
    print("임베딩 모델 로드 완료.")
except Exception as e:
    print(f"임베딩 모델 로드 중 오류 발생: {e}")
    print("pip install sentence-transformers 를 실행했는지 확인해주세요.")
    exit()

# 2. MongoDB 클라이언트 연결
client = None 
try:
    client = MongoClient(mongo_uri)
    db = client["AirBot"]
    airport_collection = db["Airport"] 
    airport_vectors_collection = db["AirportVector"]

    print("MongoDB에 성공적으로 연결되었습니다.")
    print(f"원본 컬렉션: {airport_collection.name}")
    print(f"대상 컬렉션: {airport_vectors_collection.name}")

    # 이전 실행으로 남아있는 데이터가 있다면 삭제
    user_input = input(f"'{airport_vectors_collection.name}' 컬렉션의 기존 데이터를 모두 삭제하시겠습니까? (y/n): ")
    if user_input.lower() == 'y':
        result = airport_vectors_collection.delete_many({})
        print(f"기존 {result.deleted_count}개 문서 삭제 완료.")

    print("\n'Airport' 컬렉션에서 문서를 불러와 임베딩 및 저장 중...")

    # 모든 문서 불러오기 (커서 사용)
    documents_cursor = airport_collection.find({})

    # 총 문서 개수를 미리 알아내어 tqdm으로 진행률 표시
    total_documents = airport_collection.count_documents({})
    
    processed_documents_count = 0
    documents_to_insert = []


    for doc in tqdm(documents_cursor, total=total_documents, desc="Processing Airports"):
        try:
            # 3. 임베딩할 텍스트 구성
            airport_name = doc.get('airport_name_kor', '')
            country_name = doc.get('country_name_kor', '')
            airport_code = doc.get('airport_code', '')

            # 의미를 잘 전달할 수 있는 문장 구성
            text_to_embed = f"공항 코드 {airport_code}는 {country_name}에 있는 {airport_name}입니다."

            if not text_to_embed.strip(): # 텍스트가 비어있으면 건너뛰기
                print(f"경고: _id {doc.get('_id')} 문서에서 임베딩할 텍스트를 생성할 수 없습니다. 건너_id")
                continue

            # 4. 임베딩 생성
            embedding = embedding_model.encode(text_to_embed).tolist() # NumPy 배열을 Python 리스트로 변환

            # 5. 새로운 문서 생성
            new_doc = {
                "original_id": doc['_id'], # 원본 문서의 _id를 참조용으로 저장
                "text_content": text_to_embed, # 임베딩에 사용된 원본 텍스트도 저장 (나중에 RAG에서 사용)
                "embedding": embedding # 생성된 벡터 임베딩 저장 필드
            }
            documents_to_insert.append(new_doc)
            processed_documents_count += 1

            # 6. 1000개마다 배치 삽입
            if len(documents_to_insert) >= 1000:
                airport_vectors_collection.insert_many(documents_to_insert)
                documents_to_insert = [] # 리스트 초기화

        except Exception as doc_error:
            print(f"\n문서 처리 중 오류 발생 (ID: {doc.get('_id')}): {doc_error}")
            continue # 다음 문서로 계속 진행

    # 남은 문서들 일괄 삽입
    if documents_to_insert:
        airport_vectors_collection.insert_many(documents_to_insert)

    print(f"\n총 {processed_documents_count}개의 문서를 임베딩하여 '{airport_vectors_collection.name}' 컬렉션에 저장했습니다.")

except Exception as e:
    print(f"MongoDB 작업 중 치명적인 오류 발생: {e}")

finally:
    if client:
        client.close()
        print("\nMongoDB 연결이 종료되었습니다.")