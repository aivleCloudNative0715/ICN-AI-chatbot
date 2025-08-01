# ai/chatbot/graph/utils/llm_utils.py

from typing import List
# ai/chatbot/rag/config.py에서 생성된 OpenAI 클라이언트 객체를 가져옵니다.
from ai.chatbot.rag.config import client 


def split_intents(user_input: str, supported_intents: List[str]) -> List[str]:
    """사용자 질문을 여러 개의 단일 의도 질문으로 분리합니다."""
    supported_intents_str = ", ".join(supported_intents)

    prompt = f"""
    다음 질문에 여러 의도가 포함되어 있다면, 각 의도를 정해진 목록에 있는 의도명과 매칭하여 
    하나의 독립적인 문장으로 분리해 주세요.

    <정해진 의도명 목록>
    {supported_intents_str}

    <지침>
    1. 각 의도는 독립적인 문장으로 분리하고 쉼표로 구분해 주세요.
    2. 질문의 의도가 정해진 목록에 없다면, 해당 의도를 분리하지 마세요.
    3. 질문에 단일 의도만 포함되어 있다면, 원래 질문을 그대로 반환하세요.
    
    질문: {user_input}
    응답:
    """
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "당신은 사용자의 질문 의도를 정확하게 분석하고, 복합적인 질문을 여러 개의 단일 의도로 분리하는 데 능숙한 전문가입니다. 특히 인천국제공항과 관련된 질문을 가장 잘 처리합니다."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0
    )
    
    result_text = response.choices[0].message.content
    split_questions = [q.strip() for q in result_text.split(',') if q.strip()]
    
    return split_questions

def combine_responses(responses: List[str]) -> str:
    """여러 개의 답변을 하나의 자연스러운 문장으로 결합합니다."""
    if not responses:
        return "죄송합니다. 요청하신 정보에 대한 답변을 찾을 수 없습니다."
        
    return " ".join(responses)