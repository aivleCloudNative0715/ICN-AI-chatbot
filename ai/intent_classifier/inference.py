from shared.predict_intent_and_slots import predict_top_k_intents_and_slots



# 🧪 실행 루프
if __name__ == "__main__":
    print("✨ KoBERT 기반 인텐트/슬롯 예측기입니다.")
    while True:
        text = input("\n✉️ 입력 (exit 입력 시 종료): ").strip()
        if text.lower() == "exit":
            print("👋 종료합니다.")
            break

        intents, slots = predict_top_k_intents_and_slots(text, k=3)

        print("\n🔍 예측된 인텐트 TOP 3:")
        for i, (intent, prob) in enumerate(intents, 1):
            print(f" {i}. {intent} ({prob:.4f})")

        print("\n🎯 슬롯 태깅 결과:")
        for word, slot in slots:
            print(f" - {word}: {slot}")
