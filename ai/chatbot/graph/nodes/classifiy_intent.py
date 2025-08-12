from chatbot.graph.state import ChatState
from shared.predict_intent_and_slots import predict_with_bce
from shared.config import INTENT_CLASSIFICATION


def classify_intent(state: ChatState) -> ChatState:
    # 📌 수정된 부분: ChatState에서 전체 메시지 기록을 가져옵니다.
    # get() 메서드를 사용하여 키가 없을 경우 빈 리스트를 반환해 오류를 방지합니다.
    messages = state.get("messages", [])
    
    # 📌 수정된 부분: 전체 대화 기록을 하나의 문자열로 결합합니다.
    # 대화가 처음 시작될 때는 최신 질문만 사용합니다.
    if len(messages) > 1:
        # 이전 대화 기록(AI 응답 포함)과 최신 사용자 질문을 모두 결합합니다.
        # 이렇게 하면 분류 모델이 전체 맥락을 이해할 수 있습니다.
        full_text_with_history = " ".join([m.content for m in messages])
        text_to_classify = full_text_with_history
    else:
        # 대화가 첫 번째 턴일 경우, 사용자 입력만 사용합니다.
        text_to_classify = state["user_input"]

    # 📌 수정된 부분: BCE 기반 예측 함수를 사용하여 복합 의도 감지
    result = predict_with_bce(text_to_classify, threshold=INTENT_CLASSIFICATION["DEFAULT_THRESHOLD"], top_k_intents=3)
    
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
    
    state["confidence"] = confidence
    state["top_k_intents_and_probs"] = top_k_intents_and_probs
    state["slots"] = slots
    state["is_multi_intent"] = is_multi_intent

        
    return state