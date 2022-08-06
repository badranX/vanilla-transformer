from .. import attention
import numpy as np
import torch
from matplotlib import pyplot as plt

def test_simple_test():
    x = torch.randn(3,4,5)
    out, p = attention.attention(x, x, x)

    print(p.shape)
    plt.imshow(p[0,:,:], cmap='gray')
    plt.show()
    assert out != None
