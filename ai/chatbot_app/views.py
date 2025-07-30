from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import os
import sys
from chatbot.graph.state import ChatState
from django.core.cache import cache
from chatbot.main import chat_graph

from langchain_core.messages import HumanMessage, AIMessage

# ChatState의 초기 상태를 반환하는 함수
def get_initial_state() -> ChatState:
    return {"messages": []}

# 캐시 키를 위한 상수 정의
CHATBOT_SESSION_CACHE_KEY = 'chatbot_session_{}'

class GenerateAPIView(APIView):
    """
    POST /api/generate
    챗봇 질문 요청을 처리하는 API 뷰
    """
    
    def post(self, request, *args, **kwargs):
        
        session_id = request.data.get("session_id")
        message_id = request.data.get("message_id", '') # 새로운 메시지의 ID
        parent_id = request.data.get("parent_id") # 수정/재생성일 경우 이전 메시지 ID(없으면 null)
        user_id = request.data.get("user_id")
        context = request.data.get("context", []) # context는 없으면 빈 리스트로
        
        user_message = request.data.get("content")
        
        
        # 필수 필드 누락 체크
        if not all([session_id, message_id, user_id, user_message]):
            return Response(
                {"error": "Missing required fields."},
                status=status.HTTP_400_BAD_REQUEST
            )

        if not user_message:
            return Response(
                {"error": "질문 내용(content) 필드가 필요합니다."},
                status=status.HTTP_400_BAD_REQUEST
            )        
        
        if not session_id:
            return Response(
                {"error": "session_id is required."},
                status=status.HTTP_400_BAD_REQUEST
            )
            

        # 1. 캐시에서 기존 대화 상태를 가져옴
        cache_key = CHATBOT_SESSION_CACHE_KEY.format(session_id)
        current_state = cache.get(cache_key)

        if not current_state:
            # 2. 상태가 없으면, 새로운 ChatState 객체를 생성
            print(f"디버그: 세션 ID '{session_id}'에 대한 새로운 대화 상태를 생성합니다.")
            # **(수정)** ChatState의 구조에 맞게 초기화
            current_state = get_initial_state()

        
        # 3. 사용자 질문(content)을 메시지 객체로 만들어 상태에 추가합니다.
        # **(수정)** 새로운 사용자 메시지를 기존 messages 리스트에 추가
        current_state["messages"].append(HumanMessage(content=user_message))
        current_state["user_input"] = user_message # 현재 질문도 state에 업데이트
        

        try:
            # 4. 챗봇 그래프를 실행하고, 새로운 상태를 반환
            new_state = chat_graph.invoke(current_state)


            # 5. 새로운 상태에서 응답과 메타데이터를 추출
            # 최종 답변은 LangGraph 핸들러에서 messages 리스트에 추가한 마지막 메시지
            final_message = new_state["response"]
            answer = final_message
            metadata = new_state.get("metadata", {"source": "retrieval", "confidence": 0.0})

            # 6. 업데이트된 상태를 캐시에 다시 저장
            cache.set(cache_key, new_state, timeout=1800)
            
            response_data = {
                "answer": answer,
                "metadata": metadata
            }    
            
            # response_data = {
            #     "answer": "수하물은 도착층에서 찾을 수 있습니다.",  # 실제 챗봇 응답으로 변경
            #     "metadata": {
            #         "source": "retrieval",
            #         "confidence": 0.92
            #     }
            # }
        
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
        user_id = request.data.get("user_id")
        
        # 필수 필드 누락 체크
        if not all([message_id, content, user_id]):
            return Response(
                {"error": "Missing required fields."},
                status=status.HTTP_400_BAD_REQUEST
            )

        # -------------------------------------------------------------
        # 여기에 추천 질문을 생성하는 로직을 작성합니다.
        # 예시:
        # recommend_engine = RAG_RECOMMEND_ENGINE()
        # recommended_questions = recommend_engine.get_recommendations(content)
        # -------------------------------------------------------------
        
        # 위 로직이 성공했다고 가정하고 응답 JSON을 구성합니다.
        response_data = {
            "recommended_questions": [
                "수하물 분실 시 어떻게 하나요?",
                "환승 시 수하물은 자동으로 옮겨지나요?",
                "국내선 수하물 규정은 어떻게 되나요?"
            ] # 실제 추천 질문 리스트로 변경
        }
        
        return Response(response_data, status=status.HTTP_200_OK)