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
procedure_collection = db["AirportProcedure"]

print("airportProcedure 컬렉션 데이터에 text_embedding 필드를 추가 중...")

for doc in procedure_collection.find():
    procedure_type = doc.get('procedure_type', '')
    step_number = doc.get('step_number') # 숫자는 None 체크
    sub_step = doc.get('sub_step')       # 숫자는 None 체크
    step_name = doc.get('step_name', '')
    description = doc.get('description', '')

    text_parts = []
    if procedure_type:
        text_parts.append(f"절차 유형: {procedure_type}.")
    
    # 🚨 step_number와 sub_step을 명시적으로 포함
    if step_number is not None:
        text_parts.append(f"단계 번호: {step_number}.")
    if sub_step is not None:
        text_parts.append(f"세부 단계: {sub_step}.")

    if step_name:
        text_parts.append(f"단계명: {step_name}.")
    if description:
        text_parts.append(f"설명: {description}.")

    text_to_embed = " ".join(text_parts).strip() 

    if not text_to_embed:
        print(f"경고: {doc.get('_id')} 문서에 임베딩할 유효한 텍스트가 없어 스킵합니다.")
        continue
    
    embedding = model.encode(text_to_embed).tolist()
    
    procedure_collection.update_one(
        {"_id": doc["_id"]},
        {"$set": {"text_embedding": embedding}}
    )
    print(f"'{step_name or doc.get('_id', 'Unknown')}' 문서 업데이트 완료. (유형: {procedure_type})")

print("모든 airportProcedure 문서에 text_embedding 필드 추가 완료.")