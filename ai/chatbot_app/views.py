from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework import status
import json

import hashlib
import traceback
import os
import time
import tempfile # 임시 파일을 안전하게 저장하기 위한 모듈
import fitz # PyMuPDF
from docx import Document # Python-Docx
import zipfile # ZIP 파일 처리
import xml.etree.ElementTree as ET # XML 파일 처리
from chatbot.graph.state import ChatState
from django.core.cache import cache
from chatbot.main import chat_graph
from pymongo import MongoClient
import os
from dotenv import load_dotenv

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from sentence_transformers import SentenceTransformer
from langchain_core.messages import HumanMessage, AIMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from shared.predict_intent_and_slots import predict_top_k_intents_and_slots
from chatbot.rag.utils import get_mongo_collection

from threading import Thread

COLLECTION_NAME_DEFAULT = "Cached"
VECTOR_INDEX_NAME = "cached_vector_index"
EMBEDDING_FIELD_NAME = "embedding"
TEXT_CONTENT_FIELD_NAME = "answer"


embedding_model = SentenceTransformer('dragonkue/snowflake-arctic-embed-l-v2.0-ko')


# .env 파일에서 환경변수 불러오기 (MONGO_URI)
load_dotenv()
mongo_uri = os.getenv("MONGO_URI")

client = MongoClient(mongo_uri)
db = client["AirBot"]
airport_collection = db["RecommendQuestion"] 

cached_collection = db["Cached"] # 캐시된 질문 콜렉션
cached_collection.create_index("created_at", expireAfterSeconds=300)

# ChatState의 초기 상태를 반환하는 함수
def get_initial_state() -> ChatState:
    return {"messages": []}

# 캐시 키를 위한 상수 정의
CHATBOT_SESSION_CACHE_KEY = 'chatbot_session_{}'

# 캐시된 질문을 비동기로 저장하는 함수
# 이 함수는 별도의 스레드에서 실행되어 메인 쓰레드의 블로킹을 방지
def save_embedding_async(question, answer, cached_collection, embedding_model):
    def task():
        embedding = embedding_model.encode(question).tolist()
        cached_collection.insert_one({
            "question": question,
            "embedding": embedding,
            "answer": answer,
            "created_at": datetime.now(ZoneInfo("Asia/Seoul"))
        })
    Thread(target=task).start()


# PDF, DOCX 파일에서 텍스트를 추출하는 함수
def extract_text_from_file(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        doc = fitz.open(file_path)
        text = "\n".join(page.get_text() for page in doc)
        doc.close()
        return text

    elif ext == ".docx":
        doc = Document(file_path)
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())

    elif ext == ".hwpx":
        try:
            texts = []
            
            with zipfile.ZipFile(file_path, 'r') as zipf:
                section_files = [f for f in zipf.namelist() if f.startswith('Contents/section') and f.endswith('.xml')]

                for section_file in section_files:
                    with zipf.open(section_file) as xml_file:
                        tree = ET.parse(xml_file)
                        root = tree.getroot()

                        # 네임스페이스 정의
                        ns = {
                            'hp': 'http://www.hancom.co.kr/hwpml/2011/paragraph',
                            'hs': 'http://www.hancom.co.kr/hwpml/2011/section',
                        }

                        # <hp:p> → <hp:run> → <hp:t>
                        for para in root.findall('.//hp:p', ns):
                            for run in para.findall('hp:run', ns):
                                for text_elem in run.findall('hp:t', ns):
                                    if text_elem.text:
                                        texts.append(text_elem.text.strip())

            return "\n".join(texts)
        except Exception as e:
            raise ValueError(f"HWPX 처리 오류: {str(e)}")
    
 
 # RAG 처리 함수
# 추출된 텍스트를 벡터 컬렉션에 삽입   
def perform_rag(extracted_text: str, category: str):
    # 카테고리에 따른 벡터 컬렉션 선택
    if category == "airport_info":
        target_collection = db["AirportVector"]
    elif category == "airline_info":
        target_collection = db["AirlineVector"]
    elif category == "facility_info":
        target_collection = db["AirportFacilityVector"]
    elif category == "connection_time":
        target_collection = db["ConnectionTimeVector"]
    elif category == "transit_path":
        target_collection = db["TransitPathVector"]
    elif category == "parking_lot":
        target_collection = db["ParkingLotVector"]
    elif category == "parking_lot_policy":
        target_collection = db["ParkingLotPolicyVector"]
    elif category == "airport_policy":
        target_collection = db["AirportPolicyVector"]
    else:
        raise ValueError("지원하지 않는 category.")

    # 텍스트 청크 분할
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,       # 한 청크의 최대 길이
        chunk_overlap=100,    # 문맥 유지를 위한 오버랩
        separators=["\n\n", "\n", ". ", " "]  # 문단 → 줄바꿈 → 문장 → 단어 순서로 분할
    )
    split_texts = text_splitter.split_text(extracted_text)
    embeddings = embedding_model.encode(split_texts).tolist()

    documents = [
        {"text_content": text, "embedding": vector}
        for text, vector in zip(split_texts, embeddings)
    ]

    target_collection.insert_many(documents)
    

def perform_vector_search(
    query_embedding: list,
    collection_name: str = COLLECTION_NAME_DEFAULT,
    top_k: int = 5,
    num_candidates: int = 100,
    vector_index_name: str = VECTOR_INDEX_NAME,
    embedding_field: str = EMBEDDING_FIELD_NAME,
    text_content_field: str = TEXT_CONTENT_FIELD_NAME,
    query_filter: dict = None,
    min_score: float = 0.0  # 새로 추가: 최소 점수 필터
) -> list:
    """
    MongoDB에서 벡터 검색을 수행하고, 지정된 필드의 텍스트와 점수를 반환합니다.
    """
    collection = get_mongo_collection(collection_name)

    pipeline = []

    if query_filter:
        pipeline.append({"$match": query_filter})

    pipeline.append({
        "$vectorSearch": {
            "queryVector": query_embedding,
            "path": embedding_field,
            "numCandidates": num_candidates,
            "limit": top_k,
            "index": vector_index_name
        }
    })

    pipeline.append({
        "$project": {
            "_id": 0,
            text_content_field: f"${text_content_field}",
            "score": {"$meta": "vectorSearchScore"}
        }
    })

    search_results = collection.aggregate(pipeline)

    # 점수 기반 필터링 적용
    filtered_results = [
        {"text": doc[text_content_field], "score": doc["score"]}
        for doc in search_results
        if text_content_field in doc and doc.get("score", 0.0) >= min_score
    ]

    return filtered_results



class GenerateAPIView(APIView):
    """
    POST /api/generate
    챗봇 질문 요청을 처리하는 API 뷰
    """
    
    def post(self, request, *args, **kwargs):
        
        re = 0
        
        session_id = request.data.get("session_id")
        message_id = request.data.get("message_id", '') # 새로운 메시지의 ID
        parent_id = request.data.get("parent_id") # 수정/재생성일 경우 이전 메시지 ID(없으면 null)
        
        user_message = request.data.get("content")
        
        
        # 필수 필드 누락 체크
        if not all([session_id, message_id, user_message]):
            print("DEBUG POST DATA:", request.data)
            return Response(
                {
                    "error": f"Missing required fields. "
                            f"user_message: {user_message}, "
                            f"session_id: {session_id}, "
                            f"message_id: {message_id}"
                },
                status=status.HTTP_400_BAD_REQUEST
            )

        if not user_message:
            return Response(
                {
                    "error": "질문 내용(content) 필드가 필요합니다."
                },
                status=status.HTTP_400_BAD_REQUEST
            )        
        
        if not session_id:
            return Response(
                {"error": "session_id is required."},
                status=status.HTTP_400_BAD_REQUEST
            )
            
        input_embeddings = embedding_model.encode(user_message).tolist()

        retrieved_docs = perform_vector_search(
            input_embeddings,
            collection_name="Cached",
            vector_index_name="cached_vector_index",
            query_filter={},
            top_k=1,
            min_score=0.97  # 유사도가 0.9 이상인 경우만
        )

        if retrieved_docs:
            cached_answer = retrieved_docs[0]["text"]
            similarity_score = retrieved_docs[0]["score"]
            print(f"[DEBUG] 캐시 검색 결과: {cached_answer} (유사도: {similarity_score})")
            response_data = {
                "answer": cached_answer,
                "re": re,
            }
            return Response(response_data, status=status.HTTP_200_OK)

                  

        # 1. 캐시에서 기존 대화 상태를 가져옴
        cache_key = CHATBOT_SESSION_CACHE_KEY.format(session_id)
        current_state = cache.get(cache_key)
        
        print(f"[DEBUG] 캐시 조회 - Key: {cache_key}")
        print(f"[DEBUG] 캐시 내용: {current_state}")
        

        if not current_state: 
            # 2. 상태가 없으면, 새로운 ChatState 객체를 생성
            print(f"디버그: 세션 ID '{session_id}'에 대한 새로운 대화 상태를 생성합니다.")
            # **(수정)** ChatState의 구조에 맞게 초기화
            current_state = get_initial_state()


        if parent_id and current_state.get("pre_message_id") == parent_id:
            re = 1

            
        # 3. 사용자 질문(content)을 메시지 객체로 만들어 상태에 추가합니다.
        # **(수정)** 새로운 사용자 메시지를 기존 messages 리스트에 추가
        # 만약 이전 메시지가 HumanMessage이고 마지막 메시지가 AIMessage인 경우,
        # 마지막 두 메시지를 제거하고 새로운 HumanMessage를 추가합니다.
        # 그렇지 않으면, 단순히 새로운 HumanMessage를 추가합니다.
        if re == 1:
            if (len(current_state["messages"]) >= 2 and
                isinstance(current_state["messages"][-2], HumanMessage) and
                isinstance(current_state["messages"][-1], AIMessage)):
                current_state["messages"] = current_state["messages"][:-2]
            
        # 새로운 사용자 메시지 추가
        current_state["messages"].append(HumanMessage(content=user_message))
        
        current_state["user_input"] = user_message # 현재 질문도 state에 업데이트
        current_state["rephrased_query"] = user_message
        

        try:
            # 4. 챗봇 그래프를 실행하고, 새로운 상태를 반환
            new_state = chat_graph.invoke(current_state)


            # 5. 새로운 상태에서 응답과 메타데이터를 추출
            # 최종 답변은 LangGraph 핸들러에서 messages 리스트에 추가한 마지막 메시지
            final_message = new_state["response"]
            new_state["messages"].append(AIMessage(content=final_message))
            new_state["pre_message_id"] = message_id # 현재 메시지 ID를 pre_message_id로 저장
            
            answer = final_message
            
            new_state["messages"] = new_state["messages"][-10:]

            # 6. 업데이트된 상태를 캐시에 다시 저장
            cache.set(cache_key, new_state, timeout=1800)
            
            print(f"[DEBUG] 캐시 저장 완료 - Key: {cache_key}")
            print(f"[DEBUG] 저장된 캐시 내용: {new_state}")
            
            response_data = {
                "answer": answer,
                "re": re,
                #"metadata": metadata
            }

  
                    
            excluded_phrases = [
                "죄송합니다"
            ]

            # 저장하기 전에 필터링
            if not any(phrase in answer for phrase in excluded_phrases):
                # 7. 캐시된 질문과 답변을 비동기로 저장
                save_embedding_async(user_message, answer, cached_collection, embedding_model)
            else:
                print("[DEBUG] 답변이 제외 목록에 포함되어 있어 캐시하지 않음")


            
            return Response(response_data, status=status.HTTP_200_OK)
        
        except Exception as e:
            # 챗봇 로직 내부에서 발생한 오류 처리
            print(f"챗봇 로직 실행 중 오류 발생: {e}")
            return Response(
                {"error": f"챗봇 처리 중 오류가 발생했습니다: {e}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
            
class RecommendAPIView(APIView):
    """
    POST /api/recommend
    추천 질문 요청을 처리하는 API 뷰
    """
    def post(self, request, *args, **kwargs):
        # 요청으로부터 JSON 데이터 추출
        message_id = request.data.get("message_id")
        content = request.data.get("content")
        
        # 필수 필드 누락 체크
        if not all([message_id, content]):
            return Response(
                {"error": "Missing required fields."},
                status=status.HTTP_400_BAD_REQUEST
            )
            
        intents, slots = predict_top_k_intents_and_slots(content, k=3) # 인텐트/슬롯 예측 실행
        top_intent, prob = intents[0]
        
        print(top_intent)      
        # 분류된 의도를 가진 추천 질문 문서가져오기
        matching_docs = airport_collection.find({"intent": top_intent})
        
        recommend_question = []
        for doc in matching_docs:
            questions = doc.get("recommend_question", [])
            print(questions)
            if isinstance(questions, list):
                recommend_question.extend(questions)
            elif isinstance(questions, str):
                recommend_question.append(questions)
        
        response_data = {
            "recommend_question": recommend_question
        }
        
        return Response(response_data, status=status.HTTP_200_OK)
    
class FileUploadAPIView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, format=None):
        
        category = request.data.get("category", "")
        file_obj = request.FILES.get('file')
        
        print("request.data:", request.data)
        print("request.FILES:", request.FILES)
        print("category:", category)
        print("file_obj.name:", file_obj.name if file_obj else "파일 없음")
        if not file_obj:
            return Response({"error": "No file uploaded"}, status=status.HTTP_400_BAD_REQUEST)
        
        
        # # 임시로 파일 저장(Linux/MacOS 환경에서)
        # file_path = os.path.join('/tmp', file_obj.name)
        # with open(file_path, 'wb+') as f:
        #     for chunk in file_obj.chunks():
        #         f.write(chunk)        

        # 안전한 임시 파일 저장
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file_obj.name)[1]) as tmp_file:
                for chunk in file_obj.chunks():
                    tmp_file.write(chunk)
                file_path = tmp_file.name

            # 텍스트 추출
            extracted_text = extract_text_from_file(file_path)
            if not extracted_text.strip():
                print(f"[경고] 파일에서 텍스트를 추출하지 못했습니다: {file_path}")

            # RAG 처리 (예시)
            perform_rag(extracted_text, category)

        except Exception as e:
            traceback_str = traceback.format_exc()
            print(traceback_str)  # 콘솔에 전체 스택 출력
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        finally:
            # 임시 파일 삭제
            if os.path.exists(file_path):
                os.remove(file_path)

        return Response({"message": "File processed"}, status=status.HTTP_200_OK)
