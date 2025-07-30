# ai/chatbot/rag/config.py

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

# 모든 RAG 핸들러에서 공통으로 사용할 LLM 호출 함수 (임시 구현)
# LLM 연결 전에는 검색 결과 확인용으로 사용합니다.
def common_llm_rag_caller(user_query: str, retrieved_context: str, intent_description: str) -> str:
    """
    모든 RAG 핸들러에서 재사용 가능한 LLM 호출 함수입니다.
    현재는 LLM 연결 전이므로 검색된 컨텍스트를 출력하는 방식으로 동작합니다.
    """
    if not retrieved_context.strip():
        # 검색된 정보가 없을 때의 응답
        return f"죄송합니다. 요청하신 {intent_description} 정보를 찾을 수 없습니다. 다시 질문해주시거나 다른 정보를 문의해주세요."
    
    # LLM 연결 전 테스트를 위한 출력 및 임시 응답
    print(f"\n--- [LLM으로 전달될 프롬프트 구성 예시] ---")
    print(f"**사용자 질문**: {user_query}")
    print(f"**의도**: {intent_description}")
    print(f"**검색된 컨텍스트 (상위 200자)**:\n{retrieved_context[:800]}...")
    print(f"----------------------------------------")

    # 실제 LLM API 호출 (이 부분은 나중에 실제 LLM 연동 코드로 대체됩니다)
    # prompt = f"""당신은 인천국제공항 챗봇입니다... (프롬프트 구성)"""
    # response = your_llm_client.generate(prompt)
    # return response.text

    # 현재는 검색된 내용을 바탕으로 한 임시 답변
    return f"[{intent_description} 정보] 사용자 질문 '{user_query}'에 대한 답변입니다. 검색된 내용을 바탕으로 말씀드리자면:\n{retrieved_context[:300]}..."