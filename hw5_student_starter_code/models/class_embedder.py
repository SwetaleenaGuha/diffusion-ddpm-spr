import torch
import torch.nn as nn
import math

class ClassEmbedder(nn.Module):
    def __init__(self, embed_dim, n_classes=1000, cond_drop_rate=0.1):
        super().__init__()

        # embed_dim = number of classes (e.g. 100), n_classes = embedding dim (e.g. 32)
        # index 0 is reserved as the unconditional / null token
        self.embedding = nn.Embedding(embed_dim + 1, n_classes)
        self.cond_drop_rate = cond_drop_rate
        self.num_classes = embed_dim

    def forward(self, x):
        b = x.shape[0]

        if self.cond_drop_rate > 0 and self.training:
            # randomly replace class labels with null token (index 0)
            drop_mask = torch.rand(b, device=x.device) < self.cond_drop_rate
            x = x.clone()
            x[drop_mask] = 0

        # get embedding: (N, n_classes)
        c = self.embedding(x)
        return c