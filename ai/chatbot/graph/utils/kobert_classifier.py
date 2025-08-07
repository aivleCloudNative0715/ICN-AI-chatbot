import torch
import torch.nn as nn
from transformers import BertModel, AutoTokenizer
import pickle

from shared.model import KoBERTIntentSlotModel


class KoBERTClassifier(nn.Module):
    def __init__(self, num_labels):
        super(KoBERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained("skt/kobert-base-v1")
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        return self.classifier(self.dropout(pooled_output))

