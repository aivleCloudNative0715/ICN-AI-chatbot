from graph.flow import build_chat_graph

chat_graph = build_chat_graph()

result = chat_graph.invoke({
    "user_input": "주차 요금 얼마에요?"
})

print(result["response"])
# 👉 "주차 요금 안내입니다."
