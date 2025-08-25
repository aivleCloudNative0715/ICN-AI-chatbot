from shared.config import INTENT_CLASSIFICATION
from shared.normalize_with_morph import normalize_with_morph
from shared.predict_intent_and_slots import predict_with_bce


# 🎯 라우팅 결정 함수 (3구간 임계값)
def make_routing_decision(text, tau_hi=0.8, multi_threshold=INTENT_CLASSIFICATION["DEFAULT_THRESHOLD"]):
    """
    3구간 임계값 기반 라우팅 결정

    Args:
        tau_hi: 높은 임계값 (바로 라우팅)
        multi_threshold: 복합 의도 판단 임계값
    """
    result = predict_with_bce(
        text,
        threshold=multi_threshold
    )

    max_prob = result['max_intent_prob']
    is_multi = result['is_multi_intent']

    # 복합 의도인 경우
    if is_multi:
        decision = "multi_intent"
        action = f"🧠 메인 LLM 처리: 복합 의도 ({len(result['high_confidence_intents'])}개)"
        llm_type = "main"
    # 단일 의도 + 높은 신뢰도
    elif max_prob >= tau_hi:
        decision = "route"
        top_intent = result['all_top_intents'][0][0]
        action = f"✅ 직접 라우팅: {top_intent} 핸들러 호출"
        llm_type = None
    # 단일 의도 + 낮은 신뢰도
    else:
        decision = "abstain"
        action = "🧠 메인 LLM 처리: 신뢰도 낮음, 전체 의도 분석 필요"
        llm_type = "main"

    return {
        'decision': decision,
        'action': action,
        'llm_type': llm_type,
        'confidence': max_prob,
        'intents': result['high_confidence_intents'],
        'all_intents': result['all_top_intents'],
        'slots': result['slots'],
        'is_multi_intent': is_multi
    }

# 🔍 상세 분석 함수
def analyze_prediction(text, threshold=INTENT_CLASSIFICATION["DEFAULT_THRESHOLD"], show_all_probs=False):
    """상세한 예측 분석"""
    result = predict_with_bce(text, threshold=threshold)

    print(f"\n📝 입력: {text}")
    print(f"🎯 임계값: {threshold}")
    print(f"🔢 복합 의도 여부: {'Yes' if result['is_multi_intent'] else 'No'}")

    print(f"\n🏆 임계값 이상 인텐트 ({len(result['high_confidence_intents'])}개):")
    for i, (intent, prob) in enumerate(result['high_confidence_intents'], 1):
        print(f"   {i}. {intent}: {prob:.4f}")

    print(f"\n📊 전체 Top-{len(result['all_top_intents'])} 인텐트:")
    for i, (intent, prob) in enumerate(result['all_top_intents'], 1):
        print(f"   {i}. {intent}: {prob:.4f}")

    print(f"\n🎭 슬롯 태깅 결과:")
    for word, slot in result['slots']:
        print(f"   - {word}: {slot}")

    if result['is_multi_intent']:
        print(f"\n🎯 복합 의도 감지됨!")

    return result

# 🧪 인터랙티브 테스트 함수
def interactive_test():
    """인터랙티브 테스트"""
    print("🚀 BCEWithLogitsLoss 기반 인텐트/슬롯 예측기")
    print("=" * 50)


    threshold = 0.5 # Default threshold for analyze_prediction
    multi_threshold = 0.5 # Default threshold for make_routing_decision

    while True:
        user_input = input(f"\n✉️ 입력 (Analyze Thresh={threshold:.2f}, Multi Thresh={multi_threshold:.2f}): ").strip()
        if user_input.lower() == "exit":
            print("👋 종료합니다.")
            break

        if user_input.startswith("/threshold"):
            try:
                parts = user_input.split()
                if len(parts) > 1:
                    new_threshold = float(parts[1])
                    threshold = max(0.0, min(1.0, new_threshold))
                    print(f"🎯 상세 분석 임계값 변경: {threshold:.2f}")
                if len(parts) > 2:
                    new_multi_threshold = float(parts[2])
                    multi_threshold = max(0.0, min(1.0, new_multi_threshold))
                    print(f"🎯 복합 의도 임계값 변경: {multi_threshold:.2f}")
                elif len(parts) == 2:
                    print("💡 복합 의도 임계값도 함께 변경하려면 `/threshold [분석 임계값] [복합 의도 임계값]` 형식으로 입력하세요.")

            except:
                print("❌ 사용법: /threshold [분석 임계값] [복합 의도 임계값 (선택 사항)]")
            continue

        # Process any input as a query
        if user_input:
            # Routing decision
            routing_result = make_routing_decision(user_input, multi_threshold=multi_threshold)
            print(f"\n--- 라우팅 결정 ---")
            print(f"🎯 결정: {routing_result['decision'].upper()}")
            print(f"📊 최대 신뢰도: {routing_result['confidence']:.4f}")
            print(f"🔄 액션: {routing_result['action']}")
            if routing_result['intents']:
                 intents_str = ", ".join([f"{intent}({prob:.3f})"
                                          for intent, prob in routing_result['intents']])
                 print(f"🏷️ 예측 의도 (임계값 {multi_threshold:.2f} 이상): {intents_str}")

            # Detailed analysis
            print(f"\n--- 상세 예측 분석 ---")
            analyze_prediction(
                user_input, threshold=threshold, show_all_probs=False # show_all_probs는 항상 False로 유지
            )

# 🚀 메인 실행
if __name__ == "__main__":
    # 인터랙티브 모드
    interactive_test()

    # 또는 간단한 테스트
    # intents, slots = predict_top_k_intents_and_slots("내일 비행기 시간표 알려주세요")
    # print("인텐트:", intents)
    # print("슬롯:", slots)