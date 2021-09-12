import torch
from torch import nn

from transformers.modeling_bert import BertEncoder

class MyBertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = BertEncoder(config)

    def forward(self, hidden_state, attention_mask=None):
        if attention_mask is None:
            attention_mask = torch.ones(hidden_state.size(0), hidden_state.size(1)).to(hidden_state.device)
        attention_mask = attention_mask[:, None, None, :]
        attention_mask = attention_mask.to(dtype=hidden_state.dtype)
        if attention_mask.dtype == torch.float16:
            attention_mask = (1.0 - attention_mask) * -1e4
        elif attention_mask.dtype == torch.float32:
            attention_mask = (1.0 - attention_mask) * -1e9
        head_mask = [None] * self.config.num_hidden_layers
        hidden_state, *_ = self.encoder(hidden_state, attention_mask=attention_mask, head_mask=head_mask)
        return hidden_state

class PrivateModelForIOB2Tagging(nn.Module):
    def __init__(self, config, num_labels):
        super().__init__()
        self.layer = nn.Linear(config.hidden_size, num_labels*3)

    def forward(self, hidden_state):
        return self.layer(hidden_state)
