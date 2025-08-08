import requests
from chatbot.graph.state import ChatState
from chatbot.rag.utils import get_mongo_collection
from chatbot.rag.config import client
import json

common_disclaimer = (
            "\n\n---"
            "\n주의: 이 정보는 인천국제공항 웹사이트(공식 출처)를 기반으로 제공되지만, 실제 공항 운영 정보와 다를 수 있습니다."
            "가장 정확한 최신 정보는 인천국제공항 공식 웹사이트 또는 해당 항공사/기관/시설에 직접 확인하시기 바랍니다."
        ) 

def airport_weather_current_handler(state: ChatState) -> ChatState:
    
    """
    인천공항 날씨에 대한 질문이 들어왔을 때 처리해주는 핸들러
    """
    user_query = state.get("user_input", "")
    intent_name = state.get("intent", "airport_weather_current")  # 의도 이름 명시
    
    if not user_query:
        print("디버그: 사용자 쿼리가 비어 있습니다.")
        return {**state, "response": "죄송합니다. 질문 내용을 파악할 수 없습니다. 다시 질문해주세요."}

    print(f"\n--- {intent_name.upper()} 핸들러 실행 ---")
    print(f"디버그: 사용자 쿼리 - '{user_query}'")
    
    try:
        collection_ATMOS = get_mongo_collection(collection_name="ATMOS")
        collection_TAF = get_mongo_collection(collection_name="TAF")
        
        atmos_documents = list(collection_ATMOS.find({}, {"_id": 0}))
        taf_documents = list(collection_TAF.find({}, {"_id": 0}))
        
    except Exception as e:
            error_msg = f"죄송합니다. DB 연결 또는 조회 중 오류가 발생했습니다: {e}"
            print(f"디버그: {error_msg}")
            return {**state, "response": error_msg}
    
    try: 
        prompt_template = (
            "당신은 인천국제공항의 정보를 제공하는 친절하고 유용한 챗봇입니다"
            "당신은 인천국제공항의 날씨에 대한 사용자의 질문에 대답해주어야 합니다"
            "당신이 추가적으로 참고할 수 있는 정보는 두 가지입니다."
            "{atmos_documents}에서 tm은 데이터가 측정된 시각, l_vis는 시정, ta는 0.1도 단위의 섭씨 온도, hm은 % 단위의 습도, rn은 mm단위 강수량, ws_10은 0.1m/s 단위의 10분 평균 풍속입니다"
            "{taf_documents}는 공항 예보(TAF)의 전문입니다."
            "제공받은 정보를 바탕으로, 그리고 당신이 확인 가능한 인천공항의 현 시각 날씨와 날씨 예보 정보를 기반으로 사용자의 질문에 대해서 대답하세요"
        )
        
        formatted_prompt = prompt_template.format(
            atmos_documents=json.dumps(atmos_documents, ensure_ascii=False, indent=2, default=str),
            taf_documents=json.dumps(taf_documents, ensure_ascii=False, indent=2, default=str)
        )
        
        response = client.chat.completions.create(
            model="gpt-4o-mini", # 사용할 모델 지정
            messages=[
                {"role": "system", "content": formatted_prompt},
                {"role": "user", "content": user_query}
            ],
            temperature=0.5, # 창의성 조절 (0.0은 가장 보수적, 1.0은 가장 창의적)
            max_tokens=500 # 생성할 최대 토큰 수
        )
        final_response_text = response.choices[0].message.content
        print(f"\n--- [GPT-4o-mini 응답] ---")
        print(final_response_text)

        final_response = final_response_text + common_disclaimer
        
    except Exception as e:
        print(f"디버그: 응답 처리 중 오류 발생 - {e}")
        final_response = "기상 정보를 처리하는 도중 문제가 발생했습니다. 잠시 후 다시 시도해주세요."

    return {**state, "response": final_response}