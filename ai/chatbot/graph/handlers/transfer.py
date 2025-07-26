from ai.chatbot.graph.state import ChatState
from ai.chatbot.rag.transfer_info import get_transfer_policy_info 

def transfer_info_handler(state: ChatState) -> ChatState:
    print("--- transfer_info_handler 실행 시작 ---")
    user_input = state.get("user_input", "")

    if not user_input:
        print("경고: user_input이 없습니다. 빈 응답을 반환합니다.")
        return {**state, "response": "입력된 질문이 없습니다."}

    # 🚨 get_transfer_policy_info 함수 호출
    retrieved_docs = get_transfer_policy_info(user_input) 
    
    updated_state = {
        **state, 
        "retrieved_documents": retrieved_docs,
        # 임시 응답: LLM이 아직 답변을 생성하지 않았기 때문에, 현재 상태를 알리는 메시지를 반환합니다.
        "response": f"환승 정책 정보를 검색했습니다. 다음 단계에서 답변을 생성할 예정입니다. (검색된 문서 수: {len(retrieved_docs)})"
    }
    
    print(f"--- transfer_info_handler 실행 완료. 검색된 문서 수: {len(retrieved_docs)} ---")
    return updated_state

def transfer_route_guide_handler(state: ChatState) -> ChatState:
    return {**state, "response": "환승 경로 안내입니다."}
