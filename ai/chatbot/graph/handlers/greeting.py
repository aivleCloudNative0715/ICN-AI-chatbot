from chatbot.graph.state import ChatState
from chatbot.rag.config import LLM_PROMPT_TEMPLATES

from chatbot.rag.config import client


def default_greeting_handler(state: ChatState) -> ChatState:
    """
    분류되지 않는 의도, 혹은 오류로 인해 이상한 값이 들어왔을 때의 기본 핸들러.
    사용자가 질문을 다시 할 수 있도록 유도하는 간단한 응답을 반환합니다.
    """
    user_query = state.get("user_input", "")
    intent_name = state.get("intent", "default_greeting")  # 의도 이름 명시
    
    if not user_query:
        print("디버그: 사용자 쿼리가 비어 있습니다.")
        return {**state, "response": "죄송합니다. 질문 내용을 파악할 수 없습니다. 다시 질문해주세요."}

    print(f"\n--- {intent_name.upper()} 핸들러 실행 ---")
    print(f"디버그: 사용자 쿼리 - '{user_query}'")
    
    prompt_template = """
    당신은 인천국제공항의 정보를 제공하는 친절하고 유용한 챗봇입니다. 사용자에게 반갑게 인사하고, 공항에 대한 질문을 환영하는 메시지를 작성해주세요.
    
    당신이 대답 가능한 것들은 다음과 같습니다:
    1.  공항 내의 시설 및 서비스 정보
    2.  정기 항공편 및 운항 정보
    3.  항공편명을 받을 경우 해당 항공편의 운항 정보 
    4.  항공사 정보 및 고객센터 연락처
    5.  주차장 위치 및 요금과 할인 정보
    6.  실시간 주차장 빈자리 찾기 및 주차 위치 추천
    7.  시간대 별 주차장 혼잡도 예측
    8.  공항 입국장 및 출국장 혼잡도 예측
    9.  입출국 절차 및 정책
    10.  환승 정보 및 경로 안내
    11.  수하물 규정 및 제한 물품 정보
    12.  공항 코드, 이름, 위치 등 일반 정보
    13.  인천공항 날씨

    만약 자신이 타는 항공편에 대한 구체적인 정보가 궁금하다면, 편명을 제공해 달라고 안내하세요.   
    
    사용자 질문: {user_query}


    답변:"""

    try:
        # 실제 LLM API 호출 (OpenAI gpt-4o-mini 사용)
        response = client.chat.completions.create(
            model="gpt-4o-mini", # 사용할 모델 지정
            messages=[
                {"role": "system", "content": prompt_template},
                {"role": "user", "content": user_query}
            ],
            temperature=0.5, # 창의성 조절 (0.0은 가장 보수적, 1.0은 가장 창의적)
            max_tokens=500 # 생성할 최대 토큰 수
        )
        final_response_text = response.choices[0].message.content
        print(f"\n--- [GPT-4o-mini 응답] ---")
        print(final_response_text)

        final_response = final_response_text
        
    except Exception as e:
        print(f"디버그: LLM 호출 중 오류 발생: {e}")
        # 오류 발생 시 임시 답변 또는 사용자 친화적인 메시지 반환
        return f"죄송합니다. 답변을 생성하는 중 문제가 발생했습니다. 다시 시도해 주세요. (오류: {e})"    
    
    return {**state, "response": final_response}