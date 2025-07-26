from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
import os 
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv() 

# 1. 모델 로딩 (애플리케이션 시작 시 한 번만)
_model = None
def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("dragonkue/snowflake-arctic-embed-l-v2.0-ko")
    return _model

# 2. MongoDB 클라이언트 연결 (애플리케이션 시작 시 한 번만)
_mongo_client = None
def get_mongo_client():
    global _mongo_client
    if _mongo_client is None:
        MONGO_URI = os.getenv("MONGO_URI")
        if not MONGO_URI:
            raise ValueError("MONGO_URI 환경 변수가 설정되지 않았습니다. .env 파일을 확인하세요.")
        _mongo_client = MongoClient(MONGO_URI)
    return _mongo_client

# 컬렉션을 동적으로 받아오도록 수정
def get_collection(db_name: str, collection_name: str):
    client = get_mongo_client()
    return client[db_name][collection_name]

# 🚨 인덱스 이름 매핑 딕셔너리 추가/수정
# 실제 MongoDB Atlas에 정의된 인덱스 이름을 여기에 매핑합니다.
VECTOR_INDEX_NAMES = {
    "Airline": "airline_vector_index",
    "Airport": "airport_vector_index",
    "AirportEnterprise": "aiportEnterprise_vector_index",
    "AirportProcedure": "airportProcedure_vector_index",
    "Country": "country_vector_index",
    "AirportFacility": "airportFacility_vector_index", 
}

# 3. 벡터 검색 함수 정의
# collection_name 파라미터는 Python 코드에서 사용하는 컬렉션 이름 (예: 'AirportProcedure')
def query_vector_store(collection_name: str, query_embedding: list, top_k: int = 3):
    collection = get_collection("AirBot", collection_name)

    # 🚨 인덱스 이름은 매핑 딕셔너리에서 가져옵니다.
    index_name = VECTOR_INDEX_NAMES.get(collection_name)
    if not index_name:
        raise ValueError(f"컬렉션 '{collection_name}'에 대한 벡터 인덱스 이름이 정의되지 않았습니다. VECTOR_INDEX_NAMES를 확인하세요.")

    vector_search_stage = {
        "$vectorSearch": {
            "index": index_name, # 매핑된 인덱스 이름 사용
            "path": "text_embedding",
            "queryVector": query_embedding,
            "numCandidates": 100,
            "limit": top_k
        }
    }

    pipeline = [
        vector_search_stage
    ]
    pipeline.append({"$project": {"_id": 0}}) 

    results = list(collection.aggregate(pipeline))
    return results

# --- 테스트 코드 (동일) ---
if __name__ == "__main__":
    print("--- client.py 단독 테스트 시작 ---")

    model_for_test = get_model()
    
    # Airline 컬렉션 테스트
    print("\n--- Airline 컬렉션 테스트 ---")
    query_airline = "대한항공 contact 알려줘"
    query_embedding_airline = model_for_test.encode(query_airline).tolist()
    results_airline = query_vector_store("Airline", query_embedding_airline, top_k=3) 

    if results_airline:
        print("🔍 항공사 정보 검색 결과:")
        for idx, res in enumerate(results_airline, 1):
            print(f"\n📦 결과 {idx}")
            print(f"✈️ 항공사 이름: {res.get('airline_name_kor', 'N/A')}")
            print(f"📞 연락처: {res.get('airline_contact', 'N/A')}")
            print(f"🔤 항공사 코드: {res.get('airline_code', 'N/A')}")

    # AirportProcedure 컬렉션 테스트 (새로 추가)
    print("\n--- AirportProcedure 컬렉션 테스트 ---")
    query_procedure = "입국 심사 절차 알려줘"
    query_embedding_procedure = model_for_test.encode(query_procedure).tolist()
    results_procedure = query_vector_store("AirportProcedure", query_embedding_procedure, top_k=3)

    if results_procedure:
        print("🔍 공항 절차 정보 검색 결과:")
        for idx, res in enumerate(results_procedure, 1):
            print(f"\n📦 결과 {idx}")
            print(f"📍 절차 유형: {res.get('procedure_type', 'N/A')}")
            print(f"📄 설명: {res.get('description', 'N/A')[:100]}...") # 처음 100자만 출력
            print(f"🔢 단계: {res.get('step_name', 'N/A')}")

    # 🚨 AirportFacility 컬렉션 테스트 추가
    print("\n--- AirportFacility 컬렉션 테스트 ---")
    query_facility = "약국 위치 알려줘"
    query_embedding_facility = model_for_test.encode(query_facility).tolist()
    results_facility = query_vector_store("AirportFacility", query_embedding_facility, top_k=3)

    if results_facility:
        print("🔍 공항 시설 정보 검색 결과:")
        for idx, res in enumerate(results_facility, 1):
            print(f"\n📦 결과 {idx}")
            print(f"🏢 시설 이름: {res.get('facility_name', 'N/A')}")
            print(f"🗺️ 위치: {res.get('location', 'N/A')}")
            print(f"📝 설명: {res.get('description', 'N/A')[:100]}...")
            print(f"📏 카테고리: {res.get('large_category', 'N/A')} > {res.get('medium_category', 'N/A')}")

    # 🚨 AirportEnterprise 컬렉션 테스트 추가
    print("\n--- AirportEnterprise 컬렉션 테스트 ---")
    query_enterprise = "스타벅스 운영 시간 알려줘"
    query_embedding_enterprise = model_for_test.encode(query_enterprise).tolist()
    results_enterprise = query_vector_store("AirportEnterprise", query_embedding_enterprise, top_k=3)

    if results_enterprise:
        print("🔍 공항 입점업체 정보 검색 결과:")
        for idx, res in enumerate(results_enterprise, 1):
            print(f"\n📦 결과 {idx}")
            print(f"🏪 업체 이름: {res.get('enterprise_name', 'N/A')}")
            print(f"🗺️ 위치: {res.get('location', 'N/A')}")
            print(f"⏰ 운영 시간: {res.get('service_time', 'N/A')}")
            print(f"📞 전화: {res.get('tel', 'N/A')}")

    print("\n--- client.py 단독 테스트 종료 ---")