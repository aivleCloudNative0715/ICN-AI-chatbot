# ai/chatbot/rag/utils.py

import pymongo
from pymongo.mongo_client import MongoClient # 추가 임포트
from pymongo.server_api import ServerApi     # 추가 임포트
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv
from pathlib import Path

env_path = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(dotenv_path=env_path)  # ✅ 조건 없이 수행

# 이제 안전하게 환경변수 사용 가능
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("MONGO_DB_NAME", "AirBot")
COLLECTION_NAME_DEFAULT = os.getenv("MONGO_COLLECTION_DEFAULT", "AirlineVector")
EMBEDDING_MODEL_PATH = os.getenv("EMBEDDING_MODEL_PATH", 'dragonkue/snowflake-arctic-embed-l-v2.0-ko')
VECTOR_INDEX_NAME = os.getenv("VECTOR_INDEX_NAME", "airline_vector_index")
EMBEDDING_FIELD_NAME = os.getenv("EMBEDDING_FIELD_NAME", "embedding")
TEXT_CONTENT_FIELD_NAME = os.getenv("TEXT_CONTENT_FIELD_NAME", "text_content")

# --- MongoDB 클라이언트 및 임베딩 모델 전역으로 초기화 ---
_mongo_client = None
_embedding_model = None

# get_mongo_collection 함수 대신, _mongo_client를 초기화하고 반환하는 함수를 만듭니다.
def get_mongo_client():
    """MongoDB 클라이언트 인스턴스를 반환하고, 연결 테스트를 수행합니다."""
    global _mongo_client
    if _mongo_client is None:
        if not MONGO_URI:
            raise ValueError("MONGO_URI 환경 변수가 설정되지 않았습니다. .env 파일을 확인하세요.")
        try:
            # pymongo.MongoClient 대신 MongoClient를 사용하고 server_api를 명시
            _mongo_client = MongoClient(MONGO_URI, server_api=ServerApi('1'))
            # 연결 테스트
            _mongo_client.admin.command('ping')
            print("디버그: MongoDB Atlas에 성공적으로 연결되었습니다!")
        except Exception as e:
            print(f"오류: MongoDB Atlas 연결에 실패했습니다. URI를 확인하거나, IP 접근 및 사용자 인증을 확인해주세요: {e}")
            _mongo_client = None # 연결 실패 시 클라이언트 초기화
            raise # 연결 실패 예외 발생

    return _mongo_client


def get_mongo_collection(collection_name: str = COLLECTION_NAME_DEFAULT):
    """지정된 MongoDB 컬렉션을 반환합니다."""
    # 먼저 클라이언트를 가져옵니다.
    client = get_mongo_client()
    db = client[DB_NAME]
    return db[collection_name]

def get_embedding_model():
    """임베딩 모델 인스턴스를 반환합니다."""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_PATH)
    return _embedding_model

def get_query_embedding(query: str) -> list:
    """사용자 쿼리를 벡터로 임베딩합니다."""
    model = get_embedding_model()
    return model.encode(query).tolist()

def perform_vector_search(
    query_embedding: list,
    collection_name: str = COLLECTION_NAME_DEFAULT, # 이 인자를 사용하여 특정 컬렉션 지정 가능
    top_k: int = 5,
    num_candidates: int = 100,
    vector_index_name: str = VECTOR_INDEX_NAME, # config.py에서 오버라이드될 것임
    embedding_field: str = EMBEDDING_FIELD_NAME,
    text_content_field: str = TEXT_CONTENT_FIELD_NAME,
    query_filter: dict = None # 추가적인 필터링을 위한 인자
) -> list:
    """
    MongoDB에서 벡터 검색을 수행하고, 지정된 필드의 텍스트 내용을 반환합니다.
    """
    collection = get_mongo_collection(collection_name)

    pipeline = []
    
    # query_filter가 있다면 $match 단계를 가장 먼저 추가합니다.
    if query_filter:
        pipeline.append({"$match": query_filter})

    pipeline.append(
        {
            "$vectorSearch": {
                "queryVector": query_embedding,
                "path": embedding_field,
                "numCandidates": num_candidates,
                "limit": top_k,
                "index": vector_index_name
            }
        }
    )
    pipeline.append(
        {
            "$project": {
                "_id": 0,
                text_content_field: f"${text_content_field}",
                "score": { "$meta": "vectorSearchScore" }
            }
        }
    )
    
    # 디버그: 파이프라인 출력
    # print(f"디버그: MongoDB Aggregation Pipeline for '{collection_name}': {pipeline}")

    search_results = collection.aggregate(pipeline)
    return [doc[text_content_field] for doc in search_results if text_content_field in doc]

def perform_multi_collection_search(
    query_embedding: list,
    collection_names: list, # 컬렉션 이름 리스트
    top_k_per_collection: int = 3, # 각 컬렉션당 가져올 문서 수
    **kwargs # perform_vector_search에 전달될 다른 인자들 (num_candidates, vector_index_name 등)
) -> str:
    """
    여러 MongoDB 컬렉션에서 벡터 검색을 수행하고 결과를 병합하여 반환합니다.
    """
    all_retrieved_texts = []
    for col_name in collection_names:
        print(f"디버그: '{col_name}' 컬렉션에서 검색 중...")
        # 여기서 각 컬렉션에 맞는 vector_index_name을 kwargs에 명시적으로 전달해야 할 수 있습니다.
        # RAG_SEARCH_CONFIG에서 각 컬렉션의 인덱스 이름을 가져와서 사용하도록 common_llm_rag_caller에서 처리하는 것이 더 일반적입니다.
        texts = perform_vector_search(
            query_embedding,
            collection_name=col_name,
            top_k=top_k_per_collection,
            **kwargs # 나머지 인자 전달 (여기서 vector_index_name이 올바르게 전달되는지 중요)
        )
        all_retrieved_texts.extend(texts)
    
    return "\n\n".join(all_retrieved_texts)


def close_mongo_client():
    """MongoDB 클라이언트 연결을 종료합니다."""
    global _mongo_client
    if _mongo_client:
        _mongo_client.close()
        _mongo_client = None
        print("디버그: MongoDB 클라이언트 연결이 종료되었습니다.")