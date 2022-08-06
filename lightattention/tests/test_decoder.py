from ..decoder import *
from .. import utils
from ..attention import MultiHeadAttention 
import math, copy, time
import torch
from torch import nn

def test_decoder_simple():
    N=6
    d_model=512
    d_ff=2048
    h=8
    dropout=0.1

    c = copy.deepcopy
    attn = MultiHeadAttention(h, d_model)
    ff = utils.PositionwiseFeedForward(d_model, d_ff, dropout)
    position = nn.Identity()
    
    dec = Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N)
    x = torch.randn(3, 5, d_model)

    out = dec(x, x, None, None)

    assert out != None
