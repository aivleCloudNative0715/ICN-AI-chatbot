import torch
from torch.nn.functional import softmax

from shared.load_model import model, idx2intent, idx2slot
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