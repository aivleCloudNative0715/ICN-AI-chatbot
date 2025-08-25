import torch
from torch.nn.functional import softmax, sigmoid

from shared.load_model import model, idx2intent, idx2slot, intent2idx
from shared.normalize_with_morph import normalize_with_morph
from shared.utils import tokenizer, device

# 🧱 토큰 → 단어 병합 + 슬롯 정렬
def merge_tokens_and_slots(tokens, slot_ids, idx2slot):
    merged = []
    word = ''
    current_slot = ''

    for token, slot_id in zip(tokens, slot_ids):
        slot = idx2slot.get(slot_id, 'O')

        if token in ['[CLS]', '[SEP]', '[PAD]']:
            continue

        if token.startswith("▁"):  # 새 단어 시작
            if word:
                merged.append((word, current_slot))
            word = token[1:]
            current_slot = slot
        else:
            word += token.replace("▁", "")

    if word:
        merged.append((word, current_slot))

    return merged

# 🔮 예측 함수
def predict_top_k_intents_and_slots(text, k=3):
    encoding = tokenizer(
        text,
        return_tensors='pt',
        truncation=True,
        padding='max_length',
        max_length=64
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        intent_logits, slot_logits = model(input_ids, attention_mask)
        intent_probs = softmax(intent_logits, dim=1)

        # 토큰 디버깅용
        # for i, logit in enumerate(slot_logits[0]):
        #   top_id = logit.argmax().item()
        #   top_slot = idx2slot[top_id]
        #   print(f"{i}번째 토큰 → {top_slot} ({logit[top_id].item():.4f})")

        # 🎯 인텐트 예측 (Top-k)
        topk_probs, topk_indices = torch.topk(intent_probs, k, dim=1)
        intents = [(idx2intent[idx.item()], prob.item()) for idx, prob in zip(topk_indices[0], topk_probs[0])]

        # 🎯 슬롯 예측
        slot_pred_ids = torch.argmax(slot_logits, dim=2)[0].tolist()
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

        merged = merge_tokens_and_slots(tokens, slot_pred_ids, idx2slot)

    return intents, merged

# 🔮 예측 함수 (의도 top-1만 사용)
def predict_intent(text):
    encoding = tokenizer(text, return_tensors='pt', truncation=True, padding='max_length', max_length=64)
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        intent_logits, _ = model(input_ids, attention_mask)
        intent_probs = softmax(intent_logits, dim=1)
        top_prob, top_index = torch.max(intent_probs, dim=1)
        predicted_intent = idx2intent[top_index.item()]
        return predicted_intent, top_prob.item()

# 🔮 BCEWithLogitsLoss 기반 예측 함수
def predict_with_bce(text, threshold=0.8, top_k_intents=3, max_length=64):
    """
    BCEWithLogitsLoss로 학습된 모델을 위한 예측 함수

    Args:
        text: 입력 텍스트
        threshold: Intent 분류 임계값 (default: 0.8)
        top_k_intents: 상위 K개 인텐트 반환 (default: 3)
    """
    text = normalize_with_morph(text)
    encoding = tokenizer(
        text,
        return_tensors='pt',
        truncation=True,
        padding='max_length',
        max_length=max_length
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        intent_logits, slot_logits = model(input_ids, attention_mask)

        # Intent 예측 (Sigmoid 기반)
        intent_probs = sigmoid(intent_logits)[0]  # [num_intents]

        # 임계값 이상의 인텐트들 찾기
        high_confidence_intents = []
        for i, prob in enumerate(intent_probs):
            if prob.item() >= threshold:
                intent_name = idx2intent[i]
                high_confidence_intents.append((intent_name, prob.item()))

        # 확률 순으로 정렬
        high_confidence_intents.sort(key=lambda x: x[1], reverse=True)

        # 만약 임계값 이상인 게 없다면 최고 확률 하나만
        if not high_confidence_intents:
            max_idx = torch.argmax(intent_probs).item()
            max_prob = intent_probs[max_idx].item()
            high_confidence_intents = [(idx2intent[max_idx], max_prob)]

        # Top-K 인텐트 (전체 순위용)
        topk_probs, topk_indices = torch.topk(intent_probs, min(top_k_intents, len(intent2idx)))
        all_top_intents = [(idx2intent[idx.item()], prob.item())
                          for idx, prob in zip(topk_indices, topk_probs)]

        # 슬롯 예측 (기존과 동일 - Softmax 기반)
        slot_pred_ids = torch.argmax(slot_logits, dim=2)[0].tolist()
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
        merged_slots = merge_tokens_and_slots(tokens, slot_pred_ids, idx2slot)

    return {
        'high_confidence_intents': high_confidence_intents,  # 임계값 이상
        'all_top_intents': all_top_intents,                  # 전체 Top-K
        'slots': merged_slots,
        'is_multi_intent': len(high_confidence_intents) > 1,
        'max_intent_prob': max(prob for _, prob in all_top_intents),
        'intent_probs_raw': intent_probs.cpu().numpy()
    }