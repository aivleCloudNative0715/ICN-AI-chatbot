# from ai.chatbot.graph.flow import build_chat_graph

# chat_graph = build_chat_graph()

# result = chat_graph.invoke({
#     "user_input": "제가 아이들을 데리고 제1터미널에 왔는데 단기 주차장 중에서 아이들이 좋아하는 캐릭터 샵이나 장난감 매장이 가까운 층을 추천해주실 수 있나요"
# })

# print("📌 예측 인텐트:", result.get("intent"))
# print("📌 슬롯 정보:")
# for word, slot in result.get("slots", []):
#     print(f" - {word}: {slot}")

# print("💬 응답:", result.get("response"))

# ai/chatbot/main.py

from ai.chatbot.graph.flow import build_chat_graph
from langgraph.graph import StateGraph, END
import os
import asyncio
from typing import Dict, Any

async def run_test(user_input: str):
    chat_graph = build_chat_graph()
    
    print(f"\n=======================================================")
    print(f"💬 테스트 질문: '{user_input}'")
    print(f"=======================================================")

    try:
        result: Dict[str, Any] = await chat_graph.ainvoke({"user_input": user_input})
        
        print("✅ 실행 성공")
        print("📌 예측 인텐트:", result.get("intent"))
        
        confidence = result.get("confidence")
        # --- 수정된 부분: confidence가 None이 아닐 때만 포맷팅 ---
        if confidence is not None:
            print(f"📌 최고 확신도 (Confidence): {confidence:.4f}")
        else:
            print(f"📌 최고 확신도 (Confidence): N/A")
        
        top_k_intents = result.get("top_k_intents_and_probs", [])
        if top_k_intents:
            print("📌 예측된 인텐트 TOP 3:")
            for i, (intent, prob) in enumerate(top_k_intents, 1):
                # --- 수정된 부분: prob이 None이 아닐 때만 포맷팅 ---
                if prob is not None:
                    print(f"   {i}. {intent} ({prob:.4f})")
                else:
                    print(f"   {i}. {intent} (N/A)")
        # -----------------------------------------------------------
        
        print("📌 슬롯 정보:", result.get("slots"))
        print("💬 최종 응답:", result.get("response"))

        if result.get("intent") == "complex_intent":
            print("🚀 복합 의도 처리 노드 실행됨")
        else:
            print("➡️ 단일 의도 핸들러 실행됨")

    except Exception as e:
        print("❌ 실행 실패")
        print(f"에러: {e}")

async def main():
    """
    다양한 테스트 케이스를 실행하는 메인 함수
    """
    print("✨ 챗봇 테스트를 시작합니다.")

    # 1. 단일 의도 테스트: 기존 시스템이 잘 작동하는지 확인
    await run_test("오늘 날씨 알려줘")
    await run_test("주차장 요금은 어떻게 돼?")

    # 2. 복합 의도 테스트: LLM이 질문을 잘 분리하고 답변을 종합하는지 확인
    await run_test("제1터미널 주차 요금은 어떻게 되고, 환전소 위치도 알려줘")
    await run_test("출국 절차가 궁금하고, 면세점 위치도 알려줘")

    # 3. LLM의 의도 분리 능력 테스트: 질문의 의도 수가 3개 이상일 경우
    await run_test("환전소 위치, 유아 휴게실 위치, 그리고 주차 요금은 어떻게 되는지 궁금해")

if __name__ == "__main__":
    asyncio.run(main())