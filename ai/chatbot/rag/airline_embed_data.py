from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv

# .env 파일 로드 (아래 2번 문제 해결 후)
load_dotenv() # 이 부분이 .env 파일을 로드합니다.

# 1. 모델 로딩
model = SentenceTransformer("dragonkue/snowflake-arctic-embed-l-v2.0-ko")

# 2. MongoDB 연결
# MongoDB URI를 환경 변수에서 가져옴
MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    raise ValueError("MONGO_URI 환경 변수가 설정되지 않았습니다. .env 파일을 확인하세요.")

client = MongoClient(MONGO_URI)
db = client["AirBot"]
collection = db["Airline"]

print("컬렉션 데이터에 text_embedding 필드를 추가 중...")

# 모든 항공사 문서를 순회하며 임베딩 생성 및 업데이트
for doc in collection.find():
    # 임베딩할 텍스트 조합 (항공사 이름과 연락처를 활용)
    # 필요한 경우 여기에 항공사 서비스 정보 등 더 상세한 내용을 추가하여 임베딩 텍스트를 풍부하게 만들 수 있습니다.
    
    airline_name_kor = doc.get('airline_name_kor', '')
    airline_contact = doc.get('airline_contact', '')
    airline_code = doc.get('airline_code', '')

    # 기본 텍스트
    text_parts = []
    if airline_name_kor:
        text_parts.append(f"{airline_name_kor} 항공사")
    
    # 연락처가 있을 경우에만 추가
    if airline_contact and airline_contact != 'N/A' and airline_contact.strip() != '': # 'N/A'나 공백 문자열도 확인
        text_parts.append(f"연락처는 {airline_contact}입니다.")
    
    if airline_code:
        text_parts.append(f"항공 코드는 {airline_code}입니다.")

    # 모든 파트를 조합
    text_to_embed = ". ".join(text_parts) + "." # 문장 형식으로 깔끔하게

    if not text_to_embed.strip(): # 모든 필드가 비어있어 생성된 텍스트가 없다면 스킵 또는 경고
        print(f"경고: {doc.get('_id')} 문서에 임베딩할 유효한 텍스트가 없어 스킵합니다.")
        continue
    
    # 임베딩 생성
    embedding = model.encode(text_to_embed).tolist()
    
    # 문서 업데이트 (text_embedding 필드 추가)
    collection.update_one(
        {"_id": doc["_id"]},
        {"$set": {"text_embedding": embedding}}
    )
    print(f"'{airline_name_kor or doc.get('_id', 'Unknown')}' 문서 업데이트 완료.")

print("모든 문서에 text_embedding 필드 추가 완료. 이제 검색 코드를 다시 실행해보세요.")