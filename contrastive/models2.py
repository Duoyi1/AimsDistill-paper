import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
class SentenceBERTForDownstreamTask(nn.Module):
    def __init__(self,tokenizer, model_name,dropout):
        super(SentenceBERTForDownstreamTask, self).__init__() 
        model_SB = AutoModel.from_pretrained(model_name)
        for param in model_SB.parameters():
            param.requires_grad = True
        self.m = nn.Dropout(p=dropout)
        self.bert = model_SB
        embedding_dim = model_SB.config.hidden_size

        self.classifier = nn.Linear(model_SB.config.hidden_size, 11)  # 2 for binary classification

    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        logits = torch.sigmoid(self.m(self.classifier(pooled_output)))
        return logits, pooled_output
