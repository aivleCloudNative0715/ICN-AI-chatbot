from ai.chatbot.graph.flow import build_chat_graph

chat_graph = build_chat_graph()

result = chat_graph.invoke({
    "user_input": "동물이나 축산물을 가져오면 어떻게 해야 하나요?"
})

print(result["response"])