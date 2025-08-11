# ai/chatbot/rag/llm_tools.py

import os
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path 
import json
from typing import List, Dict
from typing import Optional

# OpenAI 클라이언트를 전역으로 초기화
env_path = Path(__file__).resolve().parents[2] / ".env"
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
    
def _extract_airline_name_with_llm(user_query: str) -> Optional[str]:
    """
    LLM을 사용하여 사용자 쿼리에서 항공사 이름만 추출합니다.
    """
    system_prompt = (
        "사용자 쿼리에서 항공사 이름만 추출하여 JSON 객체로 반환해줘. "
        "정확한 항공사 이름을 찾을 수 없으면 'unknown'으로 반환해줘. "
        "응답은 반드시 `{'airline_name': '항공사 이름'}` 형식의 JSON 객체여야 해."
        "다른 설명은 추가하지 마."
    )
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query}
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        
        parsed_data = json.loads(response.choices[0].message.content)
        airline_name = parsed_data.get('airline_name', 'unknown')
        
        return airline_name if airline_name != 'unknown' else None
    
    except (json.JSONDecodeError, Exception) as e:
        print(f"디버그: LLM을 이용한 항공사 이름 추출 실패 - {e}")
        return None


def _extract_facility_names_with_llm(user_query: str) -> List[str]:
    """
    LLM을 사용하여 사용자 쿼리에서 시설 이름(명사)만 정확하게 추출합니다.
    """
    system_prompt = (
        "사용자 쿼리에서 '시설 이름'에 해당하는 명사만 정확하게 추출하여 JSON 리스트로 반환해줘. "
        "응답은 반드시 `{'facility_names': ['시설1', '시설2']}` 형식의 JSON 객체여야 해. "
        "여러 시설이 언급되면 모두 추출해줘. 조사는 제거하고 순수한 시설 이름만 추출해야 해. "
        "정확한 시설 이름을 찾을 수 없으면 빈 리스트 `[]`를 반환해줘. "
        "다른 설명은 절대 추가하지 마."
        "\n\n예시:"
        "사용자: 터미널 1에 있는 약국은 몇 시까지 해? 그리고 은행도 알려줘"
        "응답: ```json\n{\"facility_names\": [\"약국\", \"은행\"]}```"
    )
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query}
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        
        parsed_data = json.loads(response.choices[0].message.content)
        facility_names = parsed_data.get('facility_names', [])
        
        if not isinstance(facility_names, list):
            return []
            
        return [name.strip() for name in facility_names if name]
    
    except (json.JSONDecodeError, Exception) as e:
        print(f"디버그: LLM을 이용한 시설 이름 추출 실패 - {e}")
        return []

def _filter_and_rerank_docs(context: str, user_query: str) -> Optional[str]:
    """
    LLM을 사용하여 검색된 문서에서 사용자 쿼리와 가장 관련성이 높은 문서를 재정렬하고 필터링합니다.
    """
    prompt = f"""
    아래 '검색된 문서들' 중에서 '사용자 질문'과 가장 관련성이 높은 문서들만 추출해줘.
    
    <사용자 질문>
    {user_query}
    
    <검색된 문서들>
    {context}
    
    지침:
    1. '사용자 질문'에 대한 직접적인 답변을 포함하고 있는 문서들만 선택해줘.
    2. 선택된 문서들을 관련성이 높은 순서대로 나열해줘.
    3. 선택된 문서들만 그대로 반환하고, 다른 설명이나 문장은 추가하지 마.
    """
    
    messages = [
        {"role": "system", "content": "주어진 질문과 문서들 사이의 관련성을 판단하여 가장 적합한 문서를 선별하는 전문가입니다."},
        {"role": "user", "content": prompt}
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.1,
            max_tokens=1000
        )
        
        reranked_context = response.choices[0].message.content.strip()
        return reranked_context if reranked_context else None
    
    except Exception as e:
        print(f"디버그: LLM 재정렬 중 오류 발생: {e}")
        return None