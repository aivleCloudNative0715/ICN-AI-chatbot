from ai.chatbot.graph.flow import build_chat_graph

chat_graph = build_chat_graph()

result = chat_graph.invoke({
    "user_input": "제가 아이들을 데리고 제1터미널에 왔는데 단기 주차장 중에서 아이들이 좋아하는 캐릭터 샵이나 장난감 매장이 가까운 층을 추천해주실 수 있나요"
})

print("📌 예측 인텐트:", result.get("intent"))
print("📌 슬롯 정보:")
for word, slot in result.get("slots", []):
    print(f" - {word}: {slot}")

print("💬 응답:", result.get("response"))
