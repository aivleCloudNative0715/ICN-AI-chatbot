import torch.nn as nn
from transformers import BertModel

class KoBERTIntentSlotModel(nn.Module):
    def __init__(self, num_intents, num_slots):
        super().__init__()
        self.bert = BertModel.from_pretrained("skt/kobert-base-v1")
        hidden_size = self.bert.config.hidden_size

        self.intent_classifier = nn.Linear(hidden_size, num_intents)
        self.slot_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_slots)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        pooled_output = outputs.pooler_output

        intent_logits = self.intent_classifier(pooled_output)
        slot_logits = self.slot_classifier(sequence_output)

        return intent_logits, slot_logits