import math
import torch
import torch.nn.functional as F
import torch.nn as nn
import utils


def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1))/math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0
        #We assume d_v always equal d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = utils.clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        #1) Do all the linear projections in batch from d_model -> h x d_k
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1,2) for l, x in zip(self.linears, (query, key, value))]
        print("query: ", query.shape)

        #2) Apply attention
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        #3) "concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h*self.d_k)
        return self.linears[-1](x) 


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from numpy.random import default_rng
    words = 4
    S = 8
    rng = default_rng()

    mask = np.tril(np.ones([words,words], dtype= np.uint8))
    mask = torch.from_numpy(mask)
    print(mask)
    x = np.random.randn(1, words, S)
    x = torch.randn([1,words,S])
    x = np.ones([words,S]) * np.arange(0,words*S).reshape(-1, S)/(words*S)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

    m = MultiHeadAttention(words, S, 0)

    indx, indy = np.indices((words,S))
    print(indx.shape)
    x = indx + indy
    x = x/x.max()
    ax1.set_title("x input")
    ax1.imshow(x, cmap="gray")
    x = np.expand_dims(x, 0)
    x = torch.FloatTensor(x)
    print(mask)
    y, attn = attention(x, x, x, mask=mask, dropout=None)
    y = m(x,x,x)
    #y = m(x, x, x)
    print(y.shape)
    print(y.max())
    print(attn)
    ax2.set_title("attention - props on each pixel")
    ax2.imshow(np.asarray(attn.detach()).transpose(1, 2, 0), cmap="gray")
    ax3.set_title("y output")
    ax3.imshow(np.asarray(y.detach()).transpose(1,2, 0), cmap="gray")

    plt.show()
