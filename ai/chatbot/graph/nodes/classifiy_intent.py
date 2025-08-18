from chatbot.graph.state import ChatState
from shared.predict_intent_and_slots import predict_with_bce
from shared.config import INTENT_CLASSIFICATION


def classify_intent(state: ChatState) -> ChatState:
    # 📌 수정된 부분: ChatState에서 전체 메시지 기록을 가져옵니다.
    # get() 메서드를 사용하여 키가 없을 경우 빈 리스트를 반환해 오류를 방지합니다.
    messages = state.get("messages", [])
    
    # 📌 수정된 부분: 의도 분류와 슬롯 추출을 분리합니다.
    # 의도 분류: 전체 맥락 사용, 슬롯 추출: 현재 사용자 질문만 사용
    if len(messages) > 1:
        # 의도 분류를 위해 전체 대화 기록 사용
        full_text_with_history = " ".join([m.content for m in messages])
        text_to_classify = full_text_with_history
    else:
        # 대화가 첫 번째 턴일 경우, 사용자 입력만 사용
        text_to_classify = state["user_input"]

    # 📌 수정된 부분: 의도 분류와 슬롯 추출 모두 현재 질문만 사용
    # 현재 사용자의 순수한 질문만 추출
    current_user_question = messages[-1].content if messages else state["user_input"]
    
    # 의도 분류와 슬롯 추출을 한 번의 호출로 처리
    print(f"디버그: 입력 텍스트: '{current_user_question}'")
    print(f"디버그: messages 개수: {len(messages) if messages else 0}")
    if messages and len(messages) > 0:
        print(f"디버그: 마지막 메시지: '{messages[-1].content}'")
    print(f"디버그: state['user_input']: '{state.get('user_input', 'None')}'")
    
    result = predict_with_bce(current_user_question, threshold=INTENT_CLASSIFICATION["DEFAULT_THRESHOLD"], top_k_intents=3)
    
    # 결과에서 필요한 데이터 추출
    top_k_intents_and_probs = result['all_top_intents']
    high_confidence_intents = result['high_confidence_intents']
    slots = result['slots']
    is_multi_intent = result['is_multi_intent']

    if top_k_intents_and_probs:
        top_intent, confidence = top_k_intents_and_probs[0]
    else:
        top_intent, confidence = "default", 0.0

    # 복합 의도인 경우 "complex_intent"로 설정
    if is_multi_intent:
        state["intent"] = "complex_intent"
        state["detected_intents"] = high_confidence_intents
    else:
        state["intent"] = top_intent
        state["detected_intents"] = [top_k_intents_and_probs[0]]
    
    # 이전 슬롯을 백업하고 현재 슬롯을 설정
    previous_slots = state.get("slots", [])
    if previous_slots:
        state["previous_slots"] = previous_slots
        print(f"디버그: 이전 슬롯을 previous_slots에 저장: {previous_slots}")
    
    state["confidence"] = confidence
    state["top_k_intents_and_probs"] = top_k_intents_and_probs
    state["slots"] = slots  # 현재 질문에서 추출된 슬롯
    state["is_multi_intent"] = is_multi_intent
    
    print(f"디버그: 현재 질문에서 추출된 슬롯: {slots}")
    print(f"디버그: is_multi_intent = {is_multi_intent}")
    print(f"디버그: high_confidence_intents = {high_confidence_intents}")
    print(f"디버그: 임계값 이상 의도 개수: {len(high_confidence_intents)}")

        
    return state