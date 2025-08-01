from chatbot.graph.state import ChatState

def arrival_policy_info_handler(state: ChatState) -> ChatState:
    return {**state, "response": "입국 정책 정보입니다."}

def departure_policy_info_handler(state: ChatState) -> ChatState:
    return {**state, "response": "출국 정책 정보입니다."}

def baggage_claim_info_handler(state: ChatState) -> ChatState:
    return {**state, "response": "수하물 수취 정보입니다."}

def baggage_rule_query_handler(state: ChatState) -> ChatState:
    return {**state, "response": "수하물 반입 규정입니다."}
