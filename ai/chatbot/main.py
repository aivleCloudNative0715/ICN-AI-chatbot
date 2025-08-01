from ai.chatbot.graph.flow import build_chat_graph

chat_graph = build_chat_graph()

result = chat_graph.invoke({
    "user_input": "제1터미널 단기 주차장 지하 1층에 2시간 주차할 경우 주차 요금이 대략 얼마 정도 나오는지 알려주실 수 있나요"
})

print("📌 예측 인텐트:", result.get("intent"))
print("📌 슬롯 정보:")
for word, slot in result.get("slots", []):
    print(f" - {word}: {slot}")

print("💬 응답:", result.get("response"))