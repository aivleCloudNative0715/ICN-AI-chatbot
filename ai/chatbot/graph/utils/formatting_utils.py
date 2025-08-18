# formatting_utils.py - 공통 포맷팅 지침 관리
from chatbot.rag.config import DISCLAIMER, client

# 공통 포맷팅 지침 (모든 의도에서 공통으로 사용)
COMMON_FORMATTING_SUFFIX = """

다음 지침을 반드시 따르세요:
1. 모든 응답은 HTML 형식으로 작성하며, `<p>`, `<ul>`, `<li>`, `<strong>`, `<span>` 태그를 적절히 사용하세요. 마크다운 문법은 절대 사용하지 마세요.
2. 답변에서 **중요한 정보나 키워드**는 `<strong>` 태그를 사용하고, `style="color: #1976D2;"`를 적용하여 파란색으로 강조해주세요.
3. 목록을 나열할 때는 `<ul>`과 `<li>` 태그를 사용하고, `<li>` 태그에는 색상을 적용하지 마세요.
4. 답변의 시작 부분에 제목이 있다면, `<h3>` 태그를 사용하고 `style="color: #1976D2;"`를 적용하여 눈에 띄게 만들어주세요.
5. 답변 내용에 어울리는 이모지를 1-2개 포함해서 더 친근하게 만들어주세요.
6. 불필요한 서두는 생략하고, 바로 답변 본문을 시작하세요.
7. 같은 정보를 중복으로 표시하지 마세요.
8. 항공편 상태 정보는 적절한 색상으로 표시하세요 (출발: #E65100, 도착: #388E3C, 지연: #D32F2F).
9. 한국어로 답변하세요.
10. 친근하고 전문적인 톤을 유지하세요.
"""

# 의도별 추가 지침 (필요한 경우만)
INTENT_SPECIFIC_GUIDELINES = {
    "parking": """
11. 🚗 이모지를 사용하여 주차장 정보를 표시하세요.
12. 사용 가능한 자리 수와 이용률을 명확히 표시하세요.
13. 단기주차장과 장기주차장을 구분하여 표시하세요.""",

    "weather": """
11. 🌤️ 이모지를 사용하여 날씨 정보를 표시하세요.
12. 기온과 날씨 상태를 명확히 표시하세요.
13. 여행 관련 조언이 있으면 포함하세요.""",

    "baggage_claim_info": """
11. 🧳 이모지를 사용하여 수하물 정보를 표시하세요.
12. 수하물 벨트 번호를 `<strong>` 태그와 파란색으로 강조하세요.
13. 터미널별로 구분하여 표시하세요.""",

    "complex_intent": """
11. 복합 정보의 경우 섹션별로 구분하여 표시하세요.
12. 관련된 모든 정보를 체계적으로 정리하세요."""
}

def get_enhanced_prompt(original_prompt, intent_name=None):
    """
    원본 프롬프트에 포맷팅 지침을 추가해서 완전한 프롬프트 반환
    
    Args:
        original_prompt (str): 기존 시스템 프롬프트
        intent_name (str, optional): 의도명 (추가 지침이 있는 경우)
    
    Returns:
        str: 포맷팅 지침이 포함된 완전한 프롬프트
    """
    # 기본 프롬프트 + 공통 포맷팅 지침
    enhanced_prompt = original_prompt + COMMON_FORMATTING_SUFFIX
    
    # 의도별 추가 지침이 있으면 추가
    if intent_name and intent_name in INTENT_SPECIFIC_GUIDELINES:
        enhanced_prompt += INTENT_SPECIFIC_GUIDELINES[intent_name]
    
    
    return enhanced_prompt

def get_formatted_llm_response(original_prompt, user_query, intent_name=None, temperature=0.5, max_tokens=800):
    """
    포맷팅 지침이 포함된 프롬프트로 LLM을 호출하고 DISCLAIMER가 포함된 최종 응답을 반환
    
    Args:
        original_prompt (str): 기존 시스템 프롬프트
        user_query (str): 사용자 질문
        intent_name (str, optional): 의도명 (추가 지침이 있는 경우)
        temperature (float): LLM temperature 설정
        max_tokens (int): 최대 토큰 수
    
    Returns:
        str: DISCLAIMER가 포함된 최종 응답
    """
    # 포맷팅 지침이 포함된 프롬프트 생성
    enhanced_prompt = get_enhanced_prompt(original_prompt, intent_name)
    
    # LLM 호출
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": enhanced_prompt},
            {"role": "user", "content": user_query}
        ],
        temperature=temperature,
        max_tokens=max_tokens
    )
    
    final_response = response.choices[0].message.content
    
    # DISCLAIMER 추가 (complex_intent가 아닌 경우에만)
    if intent_name != "complex_intent":
        final_response += DISCLAIMER
    
    return final_response

def get_formatted_llm_response_single_message(full_prompt, intent_name=None, temperature=0.5, max_tokens=800, role="user"):
    """
    완전한 프롬프트로 LLM을 호출하고 DISCLAIMER가 포함된 최종 응답을 반환 (단일 메시지용)
    
    Args:
        full_prompt (str): 완전한 프롬프트 (이미 formatting과 user query가 포함됨)
        intent_name (str, optional): 의도명 (DISCLAIMER 추가 여부 결정)
        temperature (float): LLM temperature 설정
        max_tokens (int): 최대 토큰 수
        role (str): 메시지 역할 ("user" 또는 "system")
    
    Returns:
        str: DISCLAIMER가 포함된 최종 응답
    """
    # 포맷팅 지침 추가
    enhanced_prompt = get_enhanced_prompt(full_prompt, intent_name)
    
    # LLM 호출
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": role, "content": enhanced_prompt}
        ],
        temperature=temperature,
        max_tokens=max_tokens
    )
    
    final_response = response.choices[0].message.content
    
    # DISCLAIMER 추가 (complex_intent가 아닌 경우에만)
    if intent_name != "complex_intent":
        final_response += DISCLAIMER
    
    return final_response

