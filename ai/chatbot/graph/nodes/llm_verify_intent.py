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
    current_slots = state.get("slots", [])

    # ✅ 추가된 로직: 현재 슬롯을 다음 턴에서 사용할 수 있도록 previous_slots에 저장
    if current_slots:
        state["previous_slots"] = current_slots
    
    supported_intents_with_desc = {
        "airport_congestion_prediction": "공항 혼잡도 예측 정보",
        "default": "일반적인 인사 또는 맥락이 없는 질문",
        "facility_guide": "공항 내 시설 및 입점업체 위치/운영시간",
        "flight_info": "-3 ~ +6일 사이의 특정 항공편의 운항 정보 (출발/도착 시간, 게이트, 카운터 등)",
        "regular_schedule_query": "한 시즌의 특정 공항에 대한 정기적인 운항 스케줄",
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

    system_prompt = f"""
    당신은 의도 분류 전문가입니다. 이전 대화 기록을 참고하여 사용자의 마지막 질문에 대한 최종 의도를 판단하고, **질문을 이전 대화 맥락을 포함하여 명확하게 재구성하세요.**

    사용 가능한 의도 목록:
    {supported_intents_list_str}

    지침:
    1. **가장 중요한 지침:** 사용자의 질문이 날짜, 요일, 시간, 터미널 정보만 포함하고 있다면, **이전 대화의 의도를 그대로 유지**하세요. 새로운 의도로 변경하려면 사용자가 명확한 키워드(예: '비행기', '주차')를 언급해야 합니다.
    2. **포괄적 질문 처리:** '유의사항', '주의할 점'과 같은 포괄적 질문은 'baggage_rule_query' 또는 'immigration_policy'와 같은 관련 정책 의도로 분류하세요.
    3. '예측된 의도'가 완벽하게 일치하면, 최종 의도와 함께 재구성된 질문을 반환하세요.
    4. '예측된 의도'가 부적절하다고 판단되면, 사용 가능한 의도 목록에서 가장 적합한 의도를 찾아 새로운 의도와 함께 재구성된 질문을 반환하세요.
    5. 어떤 의도에도 해당되지 않으면, 그대로 예측된 의도와 재구성된 질문을 반환하세요.
    6. 절대 다른 설명이나 문장은 추가하지 말고, 오직 JSON 객체만 반환하세요.
    7. 재구성된 질문에는 이전 대화에서 언급된 중요한 정보(예: '인천국제공항', '혼잡도', '2터미널')를 반드시 포함하세요.

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

        if final_intent:
            print(f"디버그: LLM 검증 결과, 최종 의도: {final_intent}, 재구성된 질문: '{rephrased_query}'")
            state["intent"] = final_intent
            state["rephrased_query"] = rephrased_query
        
    except Exception as e:
        print(f"디버그: LLM 의도 검증 또는 파싱 실패 - {e}")
        pass
        
    return state