import json
from typing import Dict, Any
from openai import OpenAI
from chatbot.graph.state import ChatState
from dotenv import load_dotenv
from pathlib import Path
import os
from langchain_core.messages import HumanMessage, AIMessage

env_path = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(dotenv_path=env_path, override=True)

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

def llm_verify_intent_node(state: ChatState) -> ChatState:
    user_input = state["user_input"]
    initial_intent = state["intent"]
    messages = state.get("messages", [])
    
    supported_intents_with_desc = {
        "airport_congestion_prediction": "공항 혼잡도 예측 정보",
        "default": "일반적인 인사 또는 맥락이 없는 질문",
        "facility_guide": "공항 내 시설 및 입점업체 위치/운영시간",
        "flight_info": "특정 항공편의 운항 정보 (출발/도착 시간, 게이트, 카운터 등)",
        "regular_schedule_query": "특정 공항에 대한 정기적인 운항 스케줄",
        "airline_info_query": "항공사 고객센터 전화번호 정보",
        "airport_info": "공항 코드, 이름, 위치 등 공항 일반 정보",
        "default_greeting": "사용자가 질문을 다시 할 수 있도록 유도",
        "parking_fee_info": "주차 요금 및 할인 정책",
        "parking_congestion_prediction": "주차장 혼잡도 예측",
        "parking_location_recommendation": "주차장 위치 추천",
        "parking_availability_query": "실시간 주차 가능 대수",
        "parking_walk_time_info": "주차장-터미널 간 도보 시간 정보",
        "immigration_policy": "입출국 심사 절차, 비자, 세관 관련 정책",
        "baggage_claim_info": "수하물 찾는 곳(수취대) 정보",
        "baggage_rule_query": "수하물 반입/위탁 규정 (제한 물품 등)",
        "transfer_info": "환승 절차 및 환승 관련 정보",
        "transfer_route_guide": "환승 경로 및 최저 환승 시간",
        "airport_weather_current": "공항 현재 날씨 정보"
    }
    
    supported_intents_list_str = "\n".join(
        [f"- {k}: {v}" for k, v in supported_intents_with_desc.items() if k != "unhandled"]
    )

    # 📌 수정된 부분: 프롬프트에 재구성된 질문(rephrased_query) 반환 지시를 추가하고,
    #                JSON 응답 형식도 rephrased_query 키를 포함하도록 명시합니다.
    system_prompt = f"""
    당신은 의도 분류 전문가입니다. 이전 대화 기록을 참고하여 사용자의 마지막 질문에 대한 최종 의도를 판단하고, **질문을 이전 대화 맥락을 포함하여 명확하게 재구성하세요.**

    사용 가능한 의도 목록:
    {supported_intents_list_str}
    
    지침:
    1. '예측된 의도'가 완벽하게 일치하면, 최종 의도와 함께 재구성된 질문을 반환하세요.
    2. '예측된 의도'가 부적절하다고 판단되면, 사용 가능한 의도 목록에서 가장 적합한 의도를 찾아 새로운 의도와 함께 재구성된 질문을 반환하세요.
    3. 어떤 의도에도 해당되지 않으면, 그대로 예측된 의도와 재구성된 질문을 반환하세요.
    4. 절대 다른 설명이나 문장은 추가하지 말고, 오직 JSON 객체만 반환하세요.

    예측된 의도: {initial_intent}
    
    JSON 응답: {{"final_intent": "예시_의도명", "rephrased_query": "예시_재구성된 질문"}}
    """

    messages_for_llm = [
        {"role": "system", "content": system_prompt}
    ]
    # 전체 대화 기록을 `messages_for_llm`에 추가
    for msg in messages:
        if isinstance(msg, HumanMessage):
            messages_for_llm.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            messages_for_llm.append({"role": "assistant", "content": msg.content})

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages_for_llm,
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        
        result = response.choices[0].message.content
        parsed_result = json.loads(result)
        
        final_intent = parsed_result.get("final_intent")
        rephrased_query = parsed_result.get("rephrased_query", "")

        # 📌 수정된 부분: 재구성된 질문도 state에 저장
        if final_intent:
            print(f"디버그: LLM 검증 결과, 최종 의도: {final_intent}, 재구성된 질문: '{rephrased_query}'")
            state["intent"] = final_intent
            state["rephrased_query"] = rephrased_query
        
    except Exception as e:
        print(f"디버그: LLM 의도 검증 또는 파싱 실패 - {e}")
        pass
        
    return state