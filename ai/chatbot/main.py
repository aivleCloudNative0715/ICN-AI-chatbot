from ai.chatbot.graph.flow import build_chat_graph

chat_graph = build_chat_graph()

result = chat_graph.invoke({
    "user_input": "FX 항공사 인천공항 번호를 알려주세요"
})

print(result["response"])