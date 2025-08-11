import sys
from langchain_core.messages import HumanMessage, AIMessage
from chatbot.graph.flow import build_chat_graph

# 챗봇 그래프를 빌드합니다.
chat_graph = build_chat_graph()

# # 대화 기록을 담을 리스트를 초기 상태에 설정합니다.
# state = {
#     "messages": []
# }

# # --- 수정된 부분: 대화 루프를 추가합니다. ---
# while True:
#     # 사용자로부터 질문을 입력받습니다.
#     user_query = input("챗봇에게 질문하세요 (종료: 'exit'): ")
#     if user_query.lower() == 'exit':
#         print("챗봇을 종료합니다.")
#         break
    
#     # 챗봇에게 보낼 메시지를 생성하고 상태에 추가합니다.
#     state["user_input"] = user_query
#     state["messages"].append(HumanMessage(content=user_query))
    
#     # 그래프를 실행하고 업데이트된 상태를 받습니다.
#     # 이때, 이전 대화 기록이 담긴 state가 그대로 전달됩니다.
#     result = chat_graph.invoke(state)

#     # 챗봇의 응답을 상태에 추가합니다.
#     ai_response = result.get("response", "죄송합니다. 오류가 발생했습니다.")
#     state["messages"].append(AIMessage(content=ai_response))
    
#     # 최종 응답을 출력합니다.
#     print("\n💬 챗봇 응답:")
#     print(ai_response)
#     print("---------------------------------------")
# --- 수정된 부분 끝 ---


# import sys
# from langchain_core.messages import HumanMessage, AIMessage
# from chatbot.graph.flow import build_chat_graph

# chat_graph = build_chat_graph()

# # 사용자 질문
# user_query = ""

# # 초기 상태를 정의하고 사용자 메시지를 추가합니다.
# initial_state = {
#     "user_input": user_query,
#     "messages": [HumanMessage(content=user_query)]
# }

# # 그래프를 호출하고 결과를 받습니다.
# result = chat_graph.invoke(initial_state)

# print("📌 예측 인텐트:", result.get("intent"))
# print("📌 슬롯 정보:")
# for word, slot in result.get("slots", []):
#     print(f" - {word}: {slot}")

# print(f"📌 Top-K Intents:")
# for intent, prob in result.get('top_k_intents_and_probs'):
#     print(f"   - {intent}: {prob}", file=sys.stderr)

# print("💬 응답:", result.get("response"))
# # 챗봇의 최종 응답도 messages 리스트에 추가
# result["messages"].append(AIMessage(content=result.get("response")))

# print(result)