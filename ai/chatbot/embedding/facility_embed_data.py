# populate_facility_embeddings.py

from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# MongoDB 및 SentenceTransformer 모델 설정
MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    raise ValueError("MONGO_URI 환경 변수가 설정되지 않았습니다. .env 파일을 확인하세요.")

DB_NAME = "AirBot" # 사용할 데이터베이스 이름

# 모델 로드 (한 번만 로드)
print("SentenceTransformer 모델 로딩 중...")
try:
    model = SentenceTransformer("dragonkue/snowflake-arctic-embed-l-v2.0-ko")
    print("모델 로드 완료.")
except Exception as e:
    print(f"모델 로드 중 오류 발생: {e}")
    exit()

# MongoDB 클라이언트 연결
print("MongoDB에 연결 중...")
try:
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    print("MongoDB 연결 성공.")
except Exception as e:
    print(f"MongoDB 연결 중 오류 발생: {e}")
    exit()

def generate_and_update_embeddings(collection_name: str, text_fields: list):
    """
    지정된 컬렉션의 문서에 대해 임베딩을 생성하고 'text_embedding' 필드를 업데이트합니다.

    Args:
        collection_name (str): 임베딩을 생성할 컬렉션의 이름.
        text_fields (list): 임베딩을 생성할 때 사용할 텍스트 필드들의 리스트.
    """
    collection = db[collection_name]
    print(f"\n--- '{collection_name}' 컬렉션 임베딩 생성 및 업데이트 시작 ---")

    documents_to_update = []
    
    # 모든 문서 순회
    for doc in collection.find({}):
        # 이미 text_embedding 필드가 있는지 확인 (재실행 시 중복 작업 방지)
        if "text_embedding" in doc and doc["text_embedding"] is not None:
            # print(f"  스키핑: 문서 {doc.get('_id', 'N/A')} - 이미 text_embedding 존재.")
            continue # 이미 있으면 스킵

        # 임베딩 생성에 사용할 텍스트 조합
        combined_text = []
        for field in text_fields:
            if doc.get(field) and doc.get(field) != '-' and doc.get(field) != '':
                combined_text.append(str(doc[field]).strip())
        
        # 텍스트가 없으면 임베딩 생성 스킵
        if not combined_text:
            print(f"  경고: 문서 {doc.get('_id', 'N/A')} - 임베딩할 텍스트 필드가 비어있습니다. 스킵합니다.")
            continue

        text_for_embedding = " ".join(combined_text)
        
        try:
            # 임베딩 생성
            embedding = model.encode(text_for_embedding).tolist()
            
            # 업데이트할 문서 목록에 추가
            documents_to_update.append({
                "_id": doc["_id"],
                "text_embedding": embedding
            })
            print(f"  처리 완료: 문서 {doc.get('_id', 'N/A')}, 텍스트: '{text_for_embedding[:50]}...'")
        except Exception as e:
            print(f"  오류 발생: 문서 {doc.get('_id', 'N/A')} 임베딩 생성 실패 - {e}")
            
    if documents_to_update:
        print(f"\n총 {len(documents_to_update)}개의 문서 업데이트 예정...")
        # 일괄 업데이트
        for update_doc in documents_to_update:
            try:
                collection.update_one(
                    {"_id": update_doc["_id"]},
                    {"$set": {"text_embedding": update_doc["text_embedding"]}}
                )
            except Exception as e:
                print(f"  문서 {update_doc['_id']} 업데이트 실패: {e}")
        print(f"--- '{collection_name}' 컬렉션 업데이트 완료 ---")
    else:
        print(f"--- '{collection_name}' 컬렉션 업데이트할 문서 없음 (모두 임베딩 완료되었거나 텍스트 필드 없음) ---")

# AirportFacility 컬렉션 처리
# facility_name, location, description, large_category, medium_category, small_category를 조합
generate_and_update_embeddings(
    "AirportFacility",
    ["facility_name", "location", "description", "large_category", "medium_category", "small_category", "item"]
)

# AirportEnterprise 컬렉션 처리
# enterprise_name, location, item, service_time, tel을 조합
generate_and_update_embeddings(
    "AirportEnterprise",
    ["enterprise_name", "location", "item", "service_time", "tel"]
)

print("\n모든 임베딩 작업 완료.")
client.close()
print("MongoDB 연결 종료.")