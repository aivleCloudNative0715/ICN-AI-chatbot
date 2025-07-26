from ai.chatbot.graph.state import ChatState
from ai.chatbot.rag.arrival_policy import get_arrival_policy_info
from ai.chatbot.rag.departure_policy import get_departure_policy_info 

def arrival_policy_info_handler(state: ChatState) -> ChatState:
    print("--- arrival_policy_info_handler 실행 시작 ---")
    user_input = state.get("user_input", "")

    if not user_input:
        print("경고: user_input이 없습니다. 빈 응답을 반환합니다.")
        return {**state, "response": "입력된 질문이 없습니다."}

    retrieved_docs = get_arrival_policy_info(user_input)
    
    updated_state = {
        **state, 
        "retrieved_documents": retrieved_docs,
        "response": f"입국 정책 정보를 검색했습니다. 다음 단계에서 답변을 생성할 예정입니다. (검색된 문서 수: {len(retrieved_docs)})"
    }
    
    print(f"--- arrival_policy_info_handler 실행 완료. 검색된 문서 수: {len(retrieved_docs)} ---")
    return updated_state

def departure_policy_info_handler(state: ChatState) -> ChatState:
    print("--- departure_policy_info_handler 실행 시작 ---")
    user_input = state.get("user_input", "")

    if not user_input:
        print("경고: user_input이 없습니다. 빈 응답을 반환합니다.")
        return {**state, "response": "입력된 질문이 없습니다."}

    # 🚨 get_departure_policy_info 함수 호출
    retrieved_docs = get_departure_policy_info(user_input)
    
    updated_state = {
        **state, 
        "retrieved_documents": retrieved_docs,
        # 임시 응답: LLM이 아직 답변을 생성하지 않았기 때문에, 현재 상태를 알리는 메시지를 반환합니다.
        "response": f"출국 정책 정보를 검색했습니다. 다음 단계에서 답변을 생성할 예정입니다. (검색된 문서 수: {len(retrieved_docs)})"
    }
    
    print(f"--- departure_policy_info_handler 실행 완료. 검색된 문서 수: {len(retrieved_docs)} ---")
    return updated_state

def baggage_claim_info_handler(state: ChatState) -> ChatState:
    return {**state, "response": "수하물 수취 정보입니다."}

def baggage_rule_query_handler(state: ChatState) -> ChatState:
    return {**state, "response": "수하물 반입 규정입니다."}
