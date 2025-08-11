from chatbot.graph.state import ChatState
from shared.config import MODEL_PATH, INTENT2IDX_PATH, SLOT2IDX_PATH
from intent_classifier.inference import predict_top_k_intents_and_slots


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

    # 📌 수정된 부분: 컨텍스트가 포함된 텍스트를 분류 함수에 전달합니다.
    top_k_intents_and_probs, slots = predict_top_k_intents_and_slots(text_to_classify, k=3)

    if top_k_intents_and_probs:
        top_intent, confidence = top_k_intents_and_probs[0]
    else:
        top_intent, confidence = "default", 0.0

    state["intent"] = top_intent
    state["confidence"] = confidence
    state["top_k_intents_and_probs"] = top_k_intents_and_probs
    state["slots"] = slots
    
    return state