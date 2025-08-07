import sys

from chatbot.graph.flow import build_chat_graph

chat_graph = build_chat_graph()

result = chat_graph.invoke({
    "user_input": "tw406편 현황"
})

print("📌 예측 인텐트:", result.get("intent"))
print("📌 슬롯 정보:")
for word, slot in result.get("slots", []):
    print(f" - {word}: {slot}")

print(f"📌 Top-K Intents:")
for intent, prob in result.get('top_k_intents_and_probs'):
    print(f"  - {intent}: {prob}", file=sys.stderr)

print("💬 응답:", result.get("response"))
