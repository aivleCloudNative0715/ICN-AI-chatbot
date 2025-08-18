# formatting_utils.py - 공통 포맷팅 지침 관리

# 공통 포맷팅 지침 (모든 의도에서 공통으로 사용)
COMMON_FORMATTING_SUFFIX = """

응답 형식 가이드라인:
1. 한국어로 답변하세요.
2. 친근하고 전문적인 톤을 유지하세요.
3. 답변은 간결하면서도 필요한 정보를 모두 포함해야 합니다.
4. 적절한 이모지(🛫, 🚗, 📍 등)를 사용하여 가독성을 높이세요.
5. 중요한 정보(시간, 숫자, 상태)는 **굵은 글씨**로 강조하세요.
6. 불필요한 서두는 생략하고, 바로 답변 본문을 시작하세요.
7. 같은 정보를 중복으로 표시하지 마세요.
8. 항공편 상태 정보는 적절한 색상으로 표시하세요 (출발: #E65100, 도착: #388E3C, 지연: #D32F2F).
"""

# 의도별 추가 지침 (필요한 경우만)
INTENT_SPECIFIC_GUIDELINES = {
    "parking": """
9. 🚗 이모지를 사용하여 주차장 정보를 표시하세요.
10. 사용 가능한 자리 수와 이용률을 명확히 표시하세요.
11. 단기주차장과 장기주차장을 구분하여 표시하세요.""",

    "weather": """
9. 🌤️ 이모지를 사용하여 날씨 정보를 표시하세요.
10. 기온과 날씨 상태를 명확히 표시하세요.
11. 여행 관련 조언이 있으면 포함하세요.""",

    "baggage_claim_info": """
9. 🧳 이모지를 사용하여 수하물 정보를 표시하세요.
10. 수하물 벨트 번호를 **굵게** 강조하세요.
11. 터미널별로 구분하여 표시하세요.""",

    "complex_intent": """
9. 복합 정보의 경우 섹션별로 구분하여 표시하세요.
10. 관련된 모든 정보를 체계적으로 정리하세요."""
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

