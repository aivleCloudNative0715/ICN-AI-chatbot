# ai/chatbot/rag/llm_tools.py

import os
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path 

# OpenAI 클라이언트를 전역으로 초기화
env_path = Path(__file__).resolve().parents[3] / ".env"
load_dotenv(dotenv_path=env_path, override=True) # override=True 추가 권장

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

def extract_location_with_llm(user_query: str) -> str:
    """
    LLM을 사용하여 사용자 질문에서 공항 위치(터미널, 탑승동)를 추출합니다.
    """
    prompt = f"""
    아래 사용자 질문에서 공항의 터미널이나 위치와 관련된 정보만 추출하세요. 
    (예: 제1터미널, 제2여객터미널, 탑승동 등)
    관련 정보가 없으면 '없음'이라고 답변하세요.

    사용자 질문: {user_query}
    추출된 위치: 
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "사용자 질문에서 핵심 위치 정보만 추출하는 전문가입니다."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0
        )
        extracted_location = response.choices[0].message.content.strip()
        return extracted_location if extracted_location != '없음' else None
    except Exception as e:
        print(f"오류: LLM 위치 추출 실패 - {e}")
        return None