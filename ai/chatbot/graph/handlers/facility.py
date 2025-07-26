from ai.chatbot.graph.state import ChatState
from ai.chatbot.rag.facility_guide import get_facility_guide_info


def facility_guide_handler(state: ChatState) -> ChatState:
    print("--- facility_guide_handler 실행 시작 ---")
    user_input = state.get("user_input", "")

    if not user_input:
        print("경고: user_input이 없습니다. 빈 응답을 반환합니다.")
        return {**state, "response": "입력된 질문이 없습니다."}

    # 🚨 get_facility_guide_info 함수 호출
    retrieved_docs = get_facility_guide_info(user_input)
    
    updated_state = {
        **state, 
        "retrieved_documents": retrieved_docs,
        "response": f"시설 안내 정보를 검색했습니다. 다음 단계에서 답변을 생성할 예정입니다. (검색된 문서 수: {len(retrieved_docs)})"
    }
    
    print(f"--- facility_guide_handler 실행 완료. 검색된 문서 수: {len(retrieved_docs)} ---")
    return updated_state