from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv

load_dotenv()

model = SentenceTransformer("dragonkue/snowflake-arctic-embed-l-v2.0-ko")

MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    raise ValueError("MONGO_URI 환경 변수가 설정되지 않았습니다. .env 파일을 확인하세요.")

client = MongoClient(MONGO_URI)
db = client["AirBot"]
country_collection = db["Country"]

print("Country 컬렉션 데이터에 text_embedding 필드를 추가 중...")

for doc in country_collection.find():
    country_name_kor = doc.get('country_name_kor', '')
    visa_required = doc.get('visa_required') # boolean 타입이므로 '' 대신 None 사용
    stay_duration = doc.get('stay_duration')
    entry_requirement = doc.get('entry_requirement', '')

    text_parts = []
    if country_name_kor:
        text_parts.append(f"국가명: {country_name_kor}.")
    
    # 비자 필요 여부 (boolean)
    if visa_required is True:
        text_parts.append(f"{country_name_kor}은(는) 비자가 필요합니다.")
    elif visa_required is False:
        text_parts.append(f"{country_name_kor}은(는) 비자가 필요하지 않습니다.")
    
    # 체류 기간 (NaN 또는 숫자)
    if stay_duration is not None and not (isinstance(stay_duration, float) and stay_duration != stay_duration): # NaN 체크
        text_parts.append(f"최대 체류 기간은 {stay_duration}일입니다.")
    
    # 입국 요건 (null 또는 문자열)
    if entry_requirement: # null, 빈 문자열 모두 false로 간주
        text_parts.append(f"입국 요건: {entry_requirement}.")

    text_to_embed = " ".join(text_parts).strip() # 문장 형식으로 깔끔하게. 마지막 마침표는 필요시 추가.

    if not text_to_embed: # 모든 필드가 비어있어 생성된 텍스트가 없다면 스킵 또는 경고
        print(f"경고: {doc.get('_id')} 문서에 임베딩할 유효한 텍스트가 없어 스킵합니다.")
        continue
    
    embedding = model.encode(text_to_embed).tolist()
    
    country_collection.update_one(
        {"_id": doc["_id"]},
        {"$set": {"text_embedding": embedding}}
    )
    print(f"'{country_name_kor or doc.get('_id', 'Unknown')}' 문서 업데이트 완료.")

print("모든 Country 문서에 text_embedding 필드 추가 완료.")
print("Country_vector_index는 이미 존재하여 추가 생성할 필요가 없습니다.")