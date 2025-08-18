from chatbot.graph.state import ChatState
from chatbot.rag.utils import get_mongo_collection
from chatbot.rag.config import client
import json
from chatbot.graph.utils.formatting_utils import get_enhanced_prompt

def airport_weather_current_handler(state: ChatState) -> ChatState:
    """
    인천공항 날씨에 대한 질문이 들어왔을 때 처리해주는 핸들러
    """
    # 📌 수정된 부분: rephrased_query를 먼저 확인하고, 없으면 user_input을 사용합니다.
    query_to_process = state.get("rephrased_query") or state.get("user_input", "")
    intent_name = state.get("intent", "airport_weather_current")
    
    if not query_to_process:
        print("디버그: 사용자 쿼리가 비어 있습니다.")
        return {**state, "response": "죄송합니다. 질문 내용을 파악할 수 없습니다. 다시 질문해주세요."}

    print(f"\n--- {intent_name.upper()} 핸들러 실행 ---")
    print(f"디버그: 핸들러가 처리할 최종 쿼리 - '{query_to_process}'")
    
    # 🚀 최적화: slot에서 weather_topic 추출하여 필요한 정보만 선별
    slots = state.get("slots", [])
    weather_topics = [word for word, slot in slots if slot in ['B-weather_topic', 'I-weather_topic']]
    
    if weather_topics:
        print(f"디버그: ⚡ slot에서 날씨 주제 추출: {weather_topics}")
        # 특정 주제에 대한 최적화된 프롬프트 사용
        focused_topics = ", ".join(weather_topics)
        topic_filter = f"특히 {focused_topics}에 대한 정보를 중심으로"
    else:
        print("디버그: slot에 weather_topic 없음, 전체 날씨 정보 제공")
        topic_filter = "전반적인 날씨 정보를"
    
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
            "당신은 인천국제공항의 정보를 제공하는 친절하고 유용한 챗봇입니다."
            "당신은 인천국제공항의 날씨에 대한 사용자의 질문에 대답해주어야 합니다."
            f"{topic_filter} 답변해주세요."  # 🚀 slot 정보 활용
            "당신이 추가적으로 참고할 수 있는 정보는 두 가지입니다."
            "'{atmos_documents}'에서 tm은 데이터가 측정된 시각, l_vis는 시정, ta는 0.1도 단위의 섭씨 온도, hm은 % 단위의 습도, rn은 mm단위 강수량, ws_10은 0.1m/s 단위의 10분 평균 풍속입니다."
            "'{taf_documents}'는 공항 예보(TAF)의 전문입니다."
            "제공받은 정보를 바탕으로, 그리고 당신이 확인 가능한 인천공항의 현 시각 날씨와 날씨 예보 정보를 기반으로 사용자의 질문에 대해서 대답하세요."
        )
        
        formatted_prompt = prompt_template.format(
            atmos_documents=json.dumps(atmos_documents, ensure_ascii=False, indent=2, default=str),
            taf_documents=json.dumps(taf_documents, ensure_ascii=False, indent=2, default=str)
        )
        
        # 포맷팅 지침이 포함된 프롬프트 사용
        enhanced_prompt = get_enhanced_prompt(formatted_prompt, intent_name)
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": enhanced_prompt},
                {"role": "user", "content": query_to_process}
            ],
            temperature=0.5,
            max_tokens=600
        )
        styled_response = response.choices[0].message.content
        
    except Exception as e:
        print(f"디버그: 응답 처리 중 오류 발생 - {e}")
        styled_response = "기상 정보를 처리하는 도중 문제가 발생했습니다. 잠시 후 다시 시도해주세요."

    return {**state, "response": styled_response}