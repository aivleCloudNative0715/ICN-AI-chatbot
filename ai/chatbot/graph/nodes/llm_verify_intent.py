import json
from typing import Dict, Any
from openai import OpenAI
from chatbot.graph.state import ChatState
from dotenv import load_dotenv
from pathlib import Path
import os

env_path = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(dotenv_path=env_path, override=True)

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

def llm_verify_intent_node(state: ChatState) -> ChatState:
    user_input = state["user_input"]
    initial_intent = state["intent"]
    
    # 의도와 설명을 함께 담은 딕셔너리
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
    
    # 프롬프트에 들어갈 의도 목록 문자열 생성
    supported_intents_list_str = "\n".join(
        [f"- {k}: {v}" for k, v in supported_intents_with_desc.items() if k != "unhandled"]
    )
    supported_intents = list(supported_intents_with_desc.keys())


    prompt = f"""
    당신은 챗봇 시스템의 의도 분류 검증 도우미입니다.
    사용자 질문과 챗봇 시스템이 1차로 분류한 '예측된 의도'를 제공할 것입니다.
    당신의 임무는 '예측된 의도'가 사용자 질문에 가장 적합한 의도인지 확인하고, 만약 더 적합한 의도가 있다면 그 의도명을 JSON 형식으로 반환하는 것입니다.
    
    사용 가능한 의도 목록:
    {supported_intents_list_str}
    
    지침:
    1. '예측된 의도'가 완벽하게 일치하면, {{"final_intent": "{initial_intent}"}}를 반환하세요.
    2. '예측된 의도'가 부적절하다고 판단되면, 사용 가능한 의도 목록에서 가장 적합한 의도를 찾아 {{"final_intent": "새로운_의도명"}} 형식으로 반환하세요.
    3. 어떤 의도에도 해당되지 않으면, 그대로 {{"final_intent": "{initial_intent}"}}를 반환하세요.
    4. 절대 다른 설명이나 문장은 추가하지 말고, 오직 JSON 객체만 반환하세요.

    사용자 질문: "{user_input}"
    예측된 의도: {initial_intent}
    
    JSON 응답:"""
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "당신은 의도 분류 전문가입니다. 질문과 예측 의도를 검증하고 최종 의도명을 JSON으로 반환합니다."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        
        result = response.choices[0].message.content
        final_intent = json.loads(result)["final_intent"]
        
        print(f"디버그: LLM 검증 결과, 최종 의도: {final_intent}")
        state["intent"] = final_intent
        
    except Exception as e:
        print(f"디버그: LLM 의도 검증 또는 파싱 실패 - {e}")
        pass
        
    return state