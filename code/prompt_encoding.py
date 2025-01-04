import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


def mean_pooling(model_output: torch.Tensor, attention_mask: torch.Tensor):
    """
    Mean pooling of token embeddings, with support for attention mask
    source: "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2"
    """
    token_embeddings = model_output[0] # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


class PromptEncoder(nn.Module):
    def __init__(self, model_name: str ='sentence-transformers/all-MiniLM-L6-v2'):
        super(PromptEncoder, self).__init__()
        self.model = AutoModel.from_pretrained(model_name)

    def forward(self, encoded_input: torch.Tensor):
        model_output = self.model(**encoded_input)

        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings  # (batch_size, embedding_dim)
