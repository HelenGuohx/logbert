import torch.nn as nn


class TimeEmbedding(nn.Module):
    def __init__(self, embed_size=512):
        super().__init__()
        self.time_embed = nn.Linear(1, embed_size)

    def forward(self, time_interval):
        return self.time_embed(time_interval)
