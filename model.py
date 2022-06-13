import torch.nn as nn
import torch.nn.functional as F

import math, copy, time
import torch
import torch.nn as nn
from attention import *
import utils
from model import *
from decoder import *
from encoder import *
from torch.nn import Embedding
from batch import *


class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        encoded = self.encode(src, src_mask)
        return self.decode(encoded, src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt,  tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


class Generator(nn.Module):
    def __init__(self, d_model, vocab):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)


def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    c = copy.deepcopy
    attn = MultiHeadAttention(h, d_model)
    ff = utils.PositionwiseFeedForward(d_model, d_ff, dropout)
    position = nn.Identity()
    
    model = EncoderDecoder(
            Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout) ,N),
            Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
            nn.Sequential(utils.Embeddings(d_model, src_vocab), c(position)),
            nn.Sequential(utils.Embeddings(d_model, tgt_vocab), c(position)),
            Generator(d_model, tgt_vocab))

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model


if __name__ == "__main__":
    model = make_model(10, 10, 2)
    x = torch.rand(3,35)*10
    x = x.int()
    #mask = torch.ones(3,2, 2).int()
    mask = subsequent_mask(35)
    mask = mask.repeat(3,1,1)
    print(mask.shape)

    print(model(x, x, mask, mask ))
    print("test")
