import re
from openai import OpenAI
import os # API 키를 환경 변수에서 로드하기 위해 필요
from dotenv import load_dotenv # dotenv 라이브러리 임포트
from pathlib import Path # Path 객체 임포트
from pymongo import MongoClient
from chatbot.rag.llm_tools import _format_and_style_with_llm

env_path = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(dotenv_path=env_path, override=True) # override=True 추가 권장

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# MongoDB 클라이언트
try:
    mongo_uri = os.getenv("MONGO_URI")
    mongo_db_name = os.getenv("MONGO_DB_NAME")
    if not mongo_uri or not mongo_db_name:
        raise ValueError("MongoDB 환경 변수가 설정되지 않았습니다.")
    mongo_client = MongoClient(mongo_uri)
    mongo_client.admin.command('ping')
    print("MongoDB에 성공적으로 연결되었습니다!")
except Exception as e:
    print(f"MongoDB 연결 오류: {e}")
    mongo_client = None

# OpenAI 클라이언트
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
openai_client = OpenAI(api_key=openai_api_key)

# 다른 파일에서 불러올 변수들
db_client = mongo_client
client = openai_client
db_name = mongo_db_name


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
    "immigration_policy": {
        "collection_name": "AirportPolicyVector",
        "vector_index_name": "airportPolicy_vector_index",
        "description": "입출국 절차 및 정책",
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
    "airport_info": {
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

    "parking_fee_info": """당신은 인천국제공항 챗봇입니다. 다음 주차 요금 및 할인 정책 정보를 바탕으로 사용자 질문에 간결하고 정확하게 답변해주세요.

    다음 지침을 반드시 따르세요:
    1. 제공된 '검색된 주차 요금 및 정책' 정보 내에서만 답변하세요.
    2. 금액, 시간, 할인율 등 구체적인 숫자는 검색된 정보와 정확히 일치하게 사용하세요.
    3. **요금 계산이 필요할 경우, 다음의 엄격한 절차를 반드시 따르세요:**
        a. 사용자 질문의 총 주차 시간을 **분(minute) 단위**로 정확히 파악하세요.
        b. 검색된 정보에서 '최초 요금'에 해당하는 시간과 금액을 먼저 분리하세요.
        c. 총 시간에서 최초 요금 시간을 뺀 **'남은 시간'**을 계산하세요.
        d. 검색된 정보의 '추가 요금' 단위 시간(예: 15분)을 사용하여 남은 시간이 몇 번의 추가 요금에 해당하는지 **정확하게 계산하세요.**
        e. 최초 요금과 계산된 추가 요금을 **합산**하여 최종 금액을 도출하세요.
    4. 만약 검색된 정보만으로는 요금 계산이 불가능하거나, 사용자 질문에 대한 내용이 전혀 없을 경우에만 답변할 수 없음을 명확히 알리세요. (예: "제가 보유한 정보로는 해당 내용을 확인할 수 없습니다. 인천국제공항 공식 웹사이트에서 최신 정보를 확인하시거나, 다른 질문을 해주세요.")
    5. 사용자의 질문이 특정 주차장(예: 단기, 장기, 예약 주차장 등)에 대한 것이라면, 해당 주차장의 정보만을 바탕으로 답변하고 다른 주차장 정보를 혼용하지 마세요.

    사용자 질문: {user_query}
    검색된 주차 요금 및 정책: {retrieved_context}

    답변:""",
        
    "parking_walk_time_info": """당신은 인천국제공항 챗봇입니다. 다음 주차장-터미널 도보 시간 정보를 바탕으로 사용자 질문에 답변해주세요.
    
    다음 지침을 반드시 따르세요:
    1. 주어진 '검색된 도보 시간 정보' 내에서만 답변하세요.
    2. 도보 시간을 안내할 때는 어떤 주차장과 어떤 터미널(제1터미널, 제2터미널 등)을 기준으로 하는지 명확하게 언급해주세요. (예: "제1터미널 장기 주차장에서 도보로 약 5분 소요됩니다.")
    3. 질문이 불명확할 때는 더 자세한 위치를 알려주면 더 정확한 정보를 찾아드릴 수 있다고 답변하세요.
    4. 애매모호하거나 불필요한 정보는 추가하지 마세요.
    5. 답변은 간결하고 정확하게 작성하며, 대략적인 시간임을 명시할 수 있습니다. (예: "약 ~분", "~분 가량")
    
    사용자 질문: {user_query}
    검색된 도보 시간 정보: {retrieved_context}

    답변:""",

    "parking_location_recommendation": """당신은 인천국제공항 챗봇입니다. 다음 검색된 주차장 위치 정보를 바탕으로 사용자에게 적합한 주차장을 추천하고 상세히 안내해주세요.
    
    다음 지침을 반드시 따르세요:
    1.  제공된 '검색된 주차장 위치 정보' 내에서만 답변하고 추천하세요.
    2.  사용자의 질문 (예: "가까운", "넓은", "터미널별" 등)에 가장 부합하는 주차장을 추천하고, 그 이유를 간략하게 설명해주세요.
    3.  각 추천 주차장의 주요 특징(예: 단기/장기 여부, 터미널과의 거리, 주요 시설 근접성 등)을 간결하게 언급해주세요.
    4.  검색된 정보만으로는 사용자에게 적합한 주차장을 추천하기 어렵거나, 질문에 대한 정보가 없을 경우, 답변할 수 없음을 명확히 알리고 추가 정보를 요청하세요. (예: "어떤 종류의 주차장을 찾으시는지 (단기/장기, 특정 터미널 근처 등) 좀 더 구체적으로 알려주시면 적합한 주차장을 추천해 드릴 수 있습니다.", "제공된 정보로는 해당 내용을 추천하기 어렵습니다.")
    5.  주차 요금 정보는 직접 언급하지 마세요. (요금은 'parking_fee_info' 의도에서 다루도록 합니다.)
    6.  추천 시 주차장 간의 장단점을 직접적으로 비교하여 우위를 가르기보다는, 각 주차장의 특징을 나열하는 방식으로 정보를 제공해주세요.
    7.  마지막에 "주차 공간 현황은 실시간으로 변동될 수 있으므로, 공항 도착 전에 인천국제공항 공식 웹사이트에서 실시간 주차 정보를 확인하시는 것을 권장합니다." 와 같은 문구를 반드시 추가하여 최종 확인은 공식 출처를 통해 할 것을 안내해주세요.
    
    사용자 질문: {user_query}
    검색된 주차장 위치 정보: {retrieved_context}

    답변:""",
        
    "immigration_policy_info": """당신은 인천국제공항 정보를 제공하는 챗봇입니다. 다음 입출국 절차 및 정책 정보를 바탕으로 사용자 질문에 단계별로 상세히 답변해주세요.
    
    다음 지침을 반드시 따르세요:
    1.  제공된 '검색된 입출국 정책' 정보 내에서만 답변하세요. 없는 정보는 절대로 추론하여 답변하지 마세요.
    2.  답변은 명확하고 간결하며, 사용자 질문에 해당하는 내용에 집중하세요.
    3.  입출국 절차는 가능한 한 단계별로 나누어 설명하여 사용자가 이해하기 쉽도록 해주세요.
    4.  검색된 정보에 사용자 질문에 대한 내용이 없거나, 불충분할 경우, 답변할 수 없음을 명확히 알리세요. (예: "제가 가진 정보로는 해당 내용을 확인할 수 없습니다.")
    5.  특히 다음과 같은 정보에 대해 질문이 있을 경우, 검색된 정보에 해당 내용이 없더라도 반드시 추가적인 안내 문구를 포함하세요:
         * 비자 관련: '대한민국 국민 일반 여권 기준 정보'임을 명시하고, '관용/외교관 여권의 비자 정보는 별도로 해당 대사관 또는 기관에 직접 문의해야 한다'는 내용을 반드시 포함하세요.
         * EU 출입국 시스템(EES/ETIAS) 관련: '유럽연합(EU)의 새로운 출입국 시스템(EES, ETIAS 등)은 시행 예정이므로, 구체적인 사항은 반드시 해당 국가의 대사관 또는 관련 기관에 직접 확인해야 한다'는 내용을 포함하세요.
         * 제한 물품 관련: '제시된 제한 물품 정보는 참고 사항이며, 가장 정확하고 최신 정보는 해당 항공사, 세관 또는 관련 기관에 직접 확인해야 한다'는 내용을 반드시 포함하세요.

    사용자 질문: {user_query}
    검색된 입국 정책: {retrieved_context}

    답변:""",

    "baggage_rule_query": """당신은 인천국제공항 정보를 제공하는 챗봇입니다. 다음 수하물 규정 정보를 바탕으로 사용자 질문에 명확하게 답변해주세요.
    
    다음 지침을 반드시 따르세요:
    1.  제공된 '검색된 수하물 규정' 정보 내에서만 답변하세요. 없는 정보는 절대로 추론하여 답변하지 마세요.
    2.  답변은 명확하고 간결하며, 사용자 질문에 해당하는 내용에 집중하세요.
    3.  수하물 규정은 '기내 반입'과 '위탁 수하물'로 나누어 설명하여 사용자가 이해하기 쉽도록 해주세요.
    4.  검색된 정보에 사용자 질문에 대한 내용이 없거나, 불충분할 경우, 답변할 수 없음을 명확히 알리세요. (예: "제가 가진 정보로는 해당 내용을 확인할 수 없습니다.")
    5.  특히 '제한 물품' 관련 정보에 대한 질문이 있을 경우, 검색된 정보에 해당 내용이 없더라도 반드시 '제시된 제한 물품 정보는 참고 사항이며, 가장 정확하고 최신 정보는 해당 항공사, 세관 또는 관련 기관에 직접 확인해야 한다'는 내용을 포함하세요.
    
    사용자 질문: {user_query}
    검색된 수하물 규정: {retrieved_context}

    답변:""",

    "transfer_info": """당신은 인천국제공항 정보를 제공하는 챗봇입니다. 다음 환승 일반 정보를 바탕으로 사용자 질문에 명확하게 답변해주세요.
    
    다음 지침을 반드시 따르세요:
    1.  제공된 '검색된 환승 정보' 내에서만 답변하세요. 없는 정보는 절대로 추론하여 답변하지 마세요.
    2.  답변은 명확하고 간결하며, 사용자 질문에 해당하는 내용에 집중하세요.
    3.  환승 절차는 가능한 한 단계별로 나누어 설명하여 사용자가 이해하기 쉽도록 해주세요.
    4.  검색된 정보에 사용자 질문에 대한 내용이 없거나, 불충분할 경우, 답변할 수 없음을 명확히 알리세요. (예: "제가 가진 정보로는 해당 내용을 확인할 수 없습니다.")
    5.  환승 절차에 대한 질문일 경우, 일반적인 환승 절차(예: 환승 보안 검색, 탑승 게이트 이동 등)를 구체적으로 언급해 주세요.

    사용자 질문: {user_query}
    검색된 환승 정보: {retrieved_context}

    답변:""",

    "transfer_route_guide": """당신은 인천국제공항 정보를 제공하는 챗봇입니다. 다음 환승 경로 및 최저 환승 시간 정보를 바탕으로 사용자 질문에 상세히 답변해주세요.
    
    다음 지침을 반드시 따르세요:
    1.  제공된 '검색된 환승 경로 및 시간 정보' 내에서만 답변하세요. 없는 정보는 절대로 추론하여 답변하지 마세요.
    2.  답변은 명확하고 간결하며, 사용자 질문에 해당하는 내용에 집중하세요.
    3.  환승 경로는 가능한 한 단계별로 나누어 설명하여 사용자가 이해하기 쉽도록 해주세요.
    4.  특정 터미널(예: 제1여객터미널, 제2여객터미널) 간의 이동 시간이나 이동 방법을 구체적으로 안내하세요.
    5.  검색된 정보에 사용자 질문에 대한 내용이 없거나, 불충분할 경우, 답변할 수 없음을 명확히 알리세요. (예: "제가 가진 정보로는 해당 환승 경로에 대한 정보를 찾을 수 없습니다.")

    사용자 질문: {user_query}
    검색된 환승 경로 및 시간 정보: {retrieved_context}

    답변:""",
        
    "facility_guide": """당신은 인천국제공항 정보를 제공하는 챗봇입니다. 다음 공항 시설 및 입점 업체 정보를 바탕으로 사용자 질문에 명확하게 답변해주세요.
    
    다음 지침을 반드시 따르세요:
    1.  제공된 '검색된 시설/업체 정보' 내에서만 답변하세요. 없는 정보는 절대로 추론하여 답변하지 마세요.
    2.  답변은 명확하고 간결하며, 사용자 질문에 해당하는 내용에 집중하세요.
    3.  시설/업체의 위치, 운영 시간, 주요 서비스 등을 포함하여 구체적으로 안내하되, 제공된 정보에 없는 항목은 언급하지 마세요.
    4.  검색된 정보에 사용자 질문에 대한 내용이 없거나, 불충분할 경우, 답변할 수 없음을 명확히 알리세요. (예: "제가 가진 정보로는 해당 내용을 확인할 수 없습니다.")

    사용자 질문: {user_query}
    검색된 시설/업체 정보: {retrieved_context}

    답변:""",

    "airport_info": """
    당신은 인천국제공항 챗봇입니다. 다음 검색된 공항 일반 정보(코드, 이름, 위치 등)를 바탕으로 사용자 질문에 간결하고 정확하게 답변해주세요.
    
    다음 지침을 반드시 따르세요:
    1.  주어진 정보 내에서만 답변하세요.
    2.  검색된 정보에 사용자 질문에 대한 내용이 없을 경우, 질문 내용이 없거나 답변할 수 없음을 명확하게 알리세요. (예: "보유한 정보에는 해당 공항의 특정 정보가 없습니다.")
    3.  애매모호하거나 불필요한 정보는 추가하지 마세요.
    사용자 질문: {user_query}
    검색된 공항 정보: {retrieved_context}

    답변:""",

    # 기본 프롬프트 (정의되지 않은 의도나 오류 시 사용)
    "default": """다음 정보를 바탕으로 사용자 질문에 답변해주세요.
    사용자 질문: {user_query}
    검색된 정보: {retrieved_context}

    답변:""",

    "flight_info": """
    당신은 인천국제공항 챗봇입니다. 다음 항공편 운항 정보를 바탕으로 사용자 질문에 답변해주세요.

    다음 지침을 반드시 따르세요:
    1. 제공된 '검색된 정보' 내에서만 답변하세요. 정보가 존재한다면 반드시 그 정보를 활용하여 답변해야 합니다.
    2. 항공편명, 항공사, 출/도착 공항 정보는 명확하게 언급해주세요.
    3. 운항 날짜 정보가 있다면, 답변에 **반드시** "2025년 8월 12일"과 같이 명확히 언급해 주세요.
    4. **출발 시간 정보(`예정시간`, `변경시간`)를 확인하여 답변에 반드시 포함하세요. `예정시간`과 `변경시간`이 다를 경우 모두 언급하고, 동일할 경우 `출발 시간`으로 통일하여 보여주세요.**

    5. **LLM이 가장 우선적으로 고려해야 할 핵심 지침:**
    - 당신은 인천국제공항 챗봇입니다. 모든 답변은 **인천국제공항을 기준**으로 작성해야 합니다.
    - 제공된 '검색된 정보'의 `direction` 필드를 확인하여, 항공편이 **인천공항으로 도착**하는 편인지, **인천공항에서 출발**하는 편인지 정확하게 파악하고 답변에 반영하세요.
    - `terminal` 정보를 확인하여, `P01`은 **'제1여객터미널'**, `P03`은 **'제2여객터미널'**을 의미합니다. 답변에 터미널 정보를 반드시 포함하고, 이 명칭을 정확히 사용하세요.

    6. 만약 특정 정보(예: 게이트, 체크인 카운터)가 '정보 없음'으로 표시되면, 해당 정보가 현재 확인되지 않음을 명확히 알려주세요.
    7. 운항 현황(remark) 정보가 있을 경우, 그 내용을 간결하게 요약해서 알려주세요.
    8. 답변에 동일한 정보(예: 탑승구 번호)를 반복해서 언급하지 마세요.
    9. 검색된 정보가 있는 경우, "죄송하지만 요청하신 정보를 찾을 수 없습니다."와 같은 부정적인 답변을 절대로 생성하지 마세요.
    10. 검색된 정보에 아무것도 없을 경우, "죄송하지만 요청하신 정보를 찾을 수 없습니다."와 같이 명확하게 답변해 주세요.

    사용자 질문: {user_query}
    검색된 정보: {retrieved_context}

    답변:
    """,

    "airport_congestion_prediction": """
    당신은 인천국제공항 챗봇입니다. 다음 공항 혼잡도 예측 정보를 바탕으로 사용자 질문에 답변해주세요.
    
    다음 지침을 반드시 따르세요:
    1. 제공된 '혼잡도 예측 정보' 내에서만 답변하세요.
    2. 터미널 번호, 예상 승객 수, 혼잡도 수준을 명확하게 언급해주세요.
    3. '혼잡도 수준'은 '원활', '보통', '붐빔', '매우 붐빔' 중 하나로만 답변하세요.
    4. 제공된 정보에 사용자 질문과 관련된 내용이 없을 경우, 답변할 수 없음을 명확히 알리세요.
    5. 답변 마지막에는 반드시 다음과 같은 문구를 추가하세요: "이 정보는 예측 자료이며, 실제 상황과 다를 수 있습니다. 최신 정보는 공항 공식 안내를 확인해 주세요."
    
    사용자 질문: {user_query}
    혼잡도 예측 정보: {retrieved_context}
    
    답변:""",

    "regular_schedule_query": """
    당신은 인천국제공항 챗봇입니다. 다음 정기 운항 스케줄 정보를 바탕으로 사용자 질문에 답변해주세요.

    다음 지침을 반드시 따르세요:
    1. 제공된 '검색된 정보' 내에서만 답변하세요. 정보가 존재한다면 반드시 그 정보를 활용하여 답변해야 합니다.
    2. 항공사, 목적지 공항, 출발 시간, 운항 요일, 운항 기간 정보를 명확하게 언급해주세요.
    3. 목적지 공항의 코드(예: JFK)가 있으면, 해당 코드를 한글 공항명(예: 뉴욕/존에프케네디)으로 변환하여 함께 출력해주세요. 단, 도시명과 공항명이 중복될 경우, 가장 간결한 형태로 출력해주세요. (예: '샌프란시스코 국제공항' -> '샌프란시스코')
    4. 만약 'schedules' 리스트가 비어있다면, 해당 조건에 맞는 스케줄이 없음을 명확하게 알리세요.
    5. 운항 요일 정보(monday, tuesday 등)가 모두 true일 경우, '매일'이라고 답변하세요.
    6. 답변 마지막에는 반드시 다음과 같은 문구를 추가하세요: "이 정보는 정기 운항 스케줄이며, 실제 운항 정보와 다를 수 있습니다. 가장 정확한 정보는 해당 항공사나 인천국제공항 공식 웹사이트에서 직접 확인하시기 바랍니다."

    사용자 질문: {user_query}
    검색된 정보: {retrieved_context}

    답변:"""
}

DISCLAIMER = (
    "\n\n"
    "주의: 이 정보는 인천국제공항 웹사이트(공식 출처)를 기반으로 제공되지만, 실제 공항 운영 정보와 다를 수 있습니다."
    "가장 정확한 최신 정보는 인천국제공항 공식 웹사이트 또는 해당 항공사/기관/시설에 직접 확인하시기 바랍니다."
)

def common_llm_rag_caller(user_query: str, retrieved_context: str, intent_description: str, intent_name: str) -> str:
    """
    모든 RAG 핸들러에서 재사용 가능한 LLM 호출 함수입니다.
    의도별 맞춤 프롬프트 템플릿을 사용하여 답변을 생성합니다.
    """
    if not retrieved_context.strip():
        # 검색된 정보가 없을 때의 응답
        return f"죄송합니다. 요청하신 {intent_description} 정보를 찾을 수 없습니다. 다시 질문해주시거나 다른 정보를 문의해주세요."

    base_prompt_template = LLM_PROMPT_TEMPLATES.get(intent_name, LLM_PROMPT_TEMPLATES["default"])

    # 복합 의도일 경우 질문별 구분 지침 추가
    if intent_name == "complex_intent":
        complex_intent_instruction = (
            "\n3. 질문이 여러 개인 경우, 각 질문에 대한 답변을 명확히 구분하여 제공하세요."
            f"\n\n사용자 질문: {user_query}"
            f"\n검색된 정보: {retrieved_context}"
            f"\n\n답변:"
        )
        final_prompt = f"{base_prompt_template}{complex_intent_instruction}"
    else:
        # 단일 의도일 경우 기존 템플릿에 공통 지침만 추가
        final_prompt = (
            f"{base_prompt_template}\n\n"
            f"사용자 질문: {user_query}\n"
            f"검색된 정보: {retrieved_context}\n\n"
            f"답변:"
        )

    print("의도명 :", intent_name)
    print("\n--- LLM에 전송될 최종 프롬프트 ---")
    print(final_prompt)
    print("-----------------------------------")

    try:
        # 🚀 최적화: HTML 스타일링을 포함한 단일 LLM 호출

        html_system_prompt = (
            "당신은 인천국제공항의 정보를 제공하는 친절하고 유용한 챗봇입니다. "
            "다음 지침을 반드시 따르세요:\n"
            "1. 모든 응답은 HTML 형식으로 작성하며, `<p>`, `<ul>`, `<li>`, `<strong>`, `<span>` 태그를 적절히 사용하세요. 마크다운 문법은 절대 사용하지 마세요.\n"
            "2. 답변에서 **중요한 정보나 키워드**는 `<strong>` 태그를 사용하고, `style=\"color: #1976D2;\"`를 적용하여 파란색으로 강조해주세요.\n"
            "3. 목록을 나열할 때는 `<ul>`과 `<li>` 태그를 사용하고, `<li>` 태그에는 색상을 적용하지 마세요.\n"
            "4. 답변의 시작 부분에 제목이 있다면, `<h3>` 태그를 사용하고 `style=\"color: #1976D2;\"`를 적용하여 눈에 띄게 만들어주세요.\n"
            "5. 답변 내용에 어울리는 이모지를 1-2개 포함해서 더 친근하게 만들어주세요.\n"
            "6. 불필요한 서두는 생략하고, 바로 답변 본문을 시작하세요.\n"
            "7. 같은 정보를 중복으로 표시하지 마세요.\n"
            "8. 항공편 상태 정보는 적절한 색상으로 표시하세요 (출발: #E65100, 도착: #388E3C, 지연: #D32F2F)."
        )
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": html_system_prompt},
                {"role": "user", "content": final_prompt}
            ],
            temperature=0.5,
            max_tokens=700
        )
        
        styled_response = response.choices[0].message.content
        print(f"\n--- [최적화된 단일 LLM 응답] ---")
        print(styled_response)

        if intent_name != "complex_intent":
            styled_response += DISCLAIMER

        final_response = styled_response.replace("```html", "")
        return final_response
    
    except Exception as e:
        print(f"디버그: LLM 호출 중 오류 발생: {e}")
        return f"죄송합니다. 답변을 생성하는 중 문제가 발생했습니다. 다시 시도해 주세요. (오류: {e})"