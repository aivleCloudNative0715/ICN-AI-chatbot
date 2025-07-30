# ai/chatbot/rag/config.py
from openai import OpenAI
import os # API 키를 환경 변수에서 로드하기 위해 필요
from dotenv import load_dotenv # dotenv 라이브러리 임포트
from pathlib import Path # Path 객체 임포트

# .env 파일에서 환경 변수를 로드합니다.
# 현재 파일 (config.py)에서 최상위 디렉토리 (ICN-AI-chatbot)까지의 경로를 계산합니다.
# config.py -> rag -> chatbot -> ai -> ICN-AI-chatbot
# 따라서 .parents[3]을 사용합니다.
env_path = Path(__file__).resolve().parents[3] / ".env"
load_dotenv(dotenv_path=env_path, override=True) # override=True 추가 권장

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# 각 의도(또는 핸들러)별 RAG 검색 설정을 정의합니다.
# 이 설정은 각 핸들러에서 어떤 MongoDB 컬렉션을 사용해야 하는지 알려줍니다.
RAG_SEARCH_CONFIG = {
    "airline_info_query": {
        "collection_name": "AirlineVector",
        "vector_index_name": "airline_vector_index",
        "description": "항공사 정보 (고객센터)"
    },
    "parking_fee_info": {
        "collection_name": "ParkingLotPolicyVector",
        "vector_index_name": "parkingLotPolicy_vector_index",
        "description": "주차 요금 및 할인 정책"
    },
    "parking_walk_time_info": {
        "collection_name": "ConnectionTimeVector",
        "vector_index_name": "connectionTime_vector_index",
        "description": "주차장-터미널 도보 시간 정보"
    },
    "parking_location_recommendation": {
        "collection_name": "ParkingLotVector",
        "vector_index_name": "parkingLot_vector_index",
        "description": "주차장 위치 추천"
    },
    "arrival_policy_info": {
        "collection_name": "AirportPolicyVector",
        "vector_index_name": "airportPolicy_vector_index",
        "description": "입국 절차 및 정책",
    },
    "departure_policy_info": {
        "collection_name": "AirportPolicyVector",
        "vector_index_name": "airportPolicy_vector_index",
        "description": "출국 절차 및 정책",
    },
    "baggage_rule_query": {
        "collection_name": "AirportPolicyVector",
        "vector_index_name": "airportPolicy_vector_index",
        "description": "수하물 규정 (제한 물품 등)",
    },
    "transfer_info": {
        "collection_name": "AirportPolicyVector",
        "vector_index_name": "airportPolicy_vector_index",
        "description": "환승 일반 정보"
    },
    "transfer_route_guide": {
        "main_collection": {
            "name": "TransitPathVector",
            "vector_index": "transitPath_vector_index"
        },
        "additional_collections": [ # 추가 컬렉션 리스트
            {
                "name": "ConnectionTimeVector",
                "vector_index": "connectionTime_vector_index"
            }
        ],
        "description": "환승 경로 및 최저 환승 시간",
        "query_filter": None # 공통 필터링이 있다면 여기에 추가
    },
    "facility_guide": {
        "collection_name": "AirportFacilityVector",
        "vector_index_name": "airportFacility_vector_index",
        "description": "공항 시설 및 입점 업체 정보"
    },
    "airport_info_query": {
        "collection_name": "AirportVector",
        "vector_index_name": "airport_vector_index",
        "description": "공항 코드, 이름, 위치 등 일반 공항 정보"
    }
    # 여기에 다른 RAG 의도에 대한 설정을 추가할 수 있습니다.
}

# LLM을 위한 프롬프트 템플릿을 의도별로 정의합니다.
# 각 템플릿은 {user_query}와 {retrieved_context} 플레이스홀더를 포함해야 합니다.
LLM_PROMPT_TEMPLATES = {
    "airline_info_query": """
    당신은 인천국제공항 정보를 제공하는 챗봇입니다. 다음 검색된 항공사 정보를 바탕으로 사용자 질문에 답변해주세요.
    
    다음 지침을 반드시 따르세요:
    1.  주어진 정보 내에서만 답변하세요.
    2.  검색된 정보에 질문에 대한 내용이 없을 경우, 질문 내용이 없거나 답변할 수 없음을 명확하게 알리고, 챗봇이 제공할 수 있는 정보의 한계를 간접적으로 설명해 주세요.** (예: "해당 내용을 제공할 수 없습니다.", "현재 저의 지식 범위 내에서는 이미지를 직접 보여드리거나 이미지 관련 정보를 제공할 수 없습니다.")
    3.  '전체 목록'과 같이 검색된 정보의 범위를 넘어서는 질문일 경우, 검색된 내용의 일부임을 명확히 알리고, 사용자가 더 구체적으로 질문해야 함을 유도하세요. (예: "검색된 정보는 전체 목록의 일부일 수 있습니다. 특정 항공사에 대해 문의하시면 관련 정보를 더 정확히 찾아드릴 수 있습니다.")
    4.  답변은 명확하고 간결하게 작성하세요.
    사용자 질문: {user_query}
    검색된 항공사 정보: {retrieved_context}

    답변:""",

    "parking_fee_info": """다음 주차 요금 및 할인 정책 정보를 바탕으로 사용자 질문에 답변해주세요. 필요한 경우 관련 주차 요금 규정을 언급해주세요.
    사용자 질문: {user_query}
    검색된 주차 요금 및 정책: {retrieved_context}

    답변:""",
        
        "parking_walk_time_info": """다음 주차장-터미널 도보 시간 정보를 바탕으로 사용자 질문에 답변해주세요. 어떤 터미널 기준으로 안내해야 할지 명확히 해주세요.
    사용자 질문: {user_query}
    검색된 도보 시간 정보: {retrieved_context}

    답변:""",

        "parking_location_recommendation": """다음 주차장 위치 정보를 바탕으로 사용자에게 적합한 주차장을 추천하고 상세히 안내해주세요.
    사용자 질문: {user_query}
    검색된 주차장 위치 정보: {retrieved_context}

    답변:""",
        
        "arrival_policy_info": """다음 입국 절차 및 정책 정보를 바탕으로 사용자 질문에 단계별로 상세히 답변해주세요.
    사용자 질문: {user_query}
    검색된 입국 정책: {retrieved_context}

    답변:""",

        "departure_policy_info": """다음 출국 절차 및 정책 정보를 바탕으로 사용자 질문에 단계별로 상세히 답변해주세요. 특히 보안 심사나 출국 심사 관련 내용을 명확히 설명해주세요.
    사용자 질문: {user_query}
    검색된 출국 정책: {retrieved_context}

    답변:""",

        "baggage_rule_query": """다음 수하물 규정 정보를 바탕으로 사용자 질문에 답변해주세요. 특히 제한 물품에 대한 상세 정보를 포함해주세요.
    사용자 질문: {user_query}
    검색된 수하물 규정: {retrieved_context}

    답변:""",

        "transfer_info": """다음 환승 일반 정보를 바탕으로 사용자 질문에 명확하게 답변해주세요.
    사용자 질문: {user_query}
    검색된 환승 정보: {retrieved_context}

    답변:""",

        "transfer_route_guide": """다음 환승 경로 및 최저 환승 시간 정보를 바탕으로 사용자 질문에 상세히 답변해주세요. 특정 터미널 간의 이동 시간을 포함하여 구체적으로 안내해주세요.
    사용자 질문: {user_query}
    검색된 환승 경로 및 시간 정보: {retrieved_context}

    답변:""",
        
        "facility_guide": """다음 공항 시설 및 입점 업체 정보를 바탕으로 사용자 질문에 명확하게 답변해주세요. 위치, 운영 시간, 주요 서비스 등을 포함해주세요.
    사용자 질문: {user_query}
    검색된 시설/업체 정보: {retrieved_context}

    답변:""",

    "airport_info_query": """
    당신은 인천국제공항 챗봇입니다. 다음 검색된 공항 일반 정보(코드, 이름, 위치 등)를 바탕으로 사용자 질문에 간결하고 정확하게 답변해주세요.
    
    다음 지침을 반드시 따르세요:
    1.  주어진 정보 내에서만 답변하세요.
    2.  검색된 정보에 사용자 질문에 대한 내용이 없을 경우, 질문 내용이 없거나 답변할 수 없음을 명확하게 알리세요. (예: "제공된 정보에는 해당 공항의 특정 정보가 없습니다.")
    3.  애매모호하거나 불필요한 정보는 추가하지 마세요.
    사용자 질문: {user_query}
    검색된 공항 정보: {retrieved_context}

    답변:""",

        # 기본 프롬프트 (정의되지 않은 의도나 오류 시 사용)
        "default": """다음 정보를 바탕으로 사용자 질문에 답변해주세요.
    사용자 질문: {user_query}
    검색된 정보: {retrieved_context}

    답변:"""
}


def common_llm_rag_caller(user_query: str, retrieved_context: str, intent_description: str, intent_name: str) -> str:
    """
    모든 RAG 핸들러에서 재사용 가능한 LLM 호출 함수입니다.
    의도별 맞춤 프롬프트 템플릿을 사용하여 답변을 생성합니다.
    """
    if not retrieved_context.strip():
        # 검색된 정보가 없을 때의 응답
        return f"죄송합니다. 요청하신 {intent_description} 정보를 찾을 수 없습니다. 다시 질문해주시거나 다른 정보를 문의해주세요."
    
    # 의도에 맞는 프롬프트 템플릿을 가져오거나, 없으면 'default' 템플릿 사용
    prompt_template = LLM_PROMPT_TEMPLATES.get(intent_name, LLM_PROMPT_TEMPLATES["default"])
    
    # 프롬프트 구성
    final_prompt = prompt_template.format(user_query=user_query, retrieved_context=retrieved_context)

    # --- 추가: LLM에 전송될 최종 프롬프트 출력 ---
    print("\n--- LLM에 전송될 최종 프롬프트 ---")
    print(final_prompt)
    print("-----------------------------------")
    # --- 여기까지 추가 ---

    try:
        # 실제 LLM API 호출 (OpenAI gpt-4o-mini 사용)
        response = client.chat.completions.create(
            model="gpt-4o-mini", # 사용할 모델 지정
            messages=[
                {"role": "system", "content": "당신은 인천국제공항의 정보를 제공하는 친절하고 유용한 챗봇입니다."},
                {"role": "user", "content": final_prompt}
            ],
            temperature=0.5, # 창의성 조절 (0.0은 가장 보수적, 1.0은 가장 창의적)
            max_tokens=500 # 생성할 최대 토큰 수
        )
        final_response_text = response.choices[0].message.content
        print(f"\n--- [GPT-4o-mini 응답] ---")
        print(final_response_text)
        print(f"--------------------------")

        # 모든 답변에 공통적으로 추가될 주의 문구
        common_disclaimer = (
            "\n\n---"
            "\n주의: 이 정보는 인천국제공항 웹사이트(공식 출처)를 기반으로 제공되지만, 실제 공항 운영 정보와 다를 수 있습니다."
            "가장 정확한 최신 정보는 인천국제공항 공식 웹사이트 또는 해당 항공사/기관/시설에 직접 확인하시기 바랍니다."
        )

        return final_response_text + common_disclaimer

    except Exception as e:
        print(f"디버그: LLM 호출 중 오류 발생: {e}")
        # 오류 발생 시 임시 답변 또는 사용자 친화적인 메시지 반환
        return f"죄송합니다. 답변을 생성하는 중 문제가 발생했습니다. 다시 시도해 주세요. (오류: {e})"