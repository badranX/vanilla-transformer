from ..encoder import *
from .. import utils
from ..attention import MultiHeadAttention 
import math, copy, time
import torch
from torch import nn


def test_encoder_simple():
    N=6
    d_model=512
    d_ff=2048
    h=8
    dropout=0.1

    c = copy.deepcopy
    attn = MultiHeadAttention(h, d_model)
    ff = utils.PositionwiseFeedForward(d_model, d_ff, dropout)
    position = nn.Identity()
    
    enc = Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout) ,N)

    x = torch.randn(3, 5, d_model)

    out = enc(x, None)

    assert out != None
