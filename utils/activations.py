# Activation functions

import jittor as jt 
from jittor import nn

# SiLU https://arxiv.org/pdf/1606.08415.pdf ----------------------------------------------------------------------------
class SiLU(nn.Module):  # export-friendly version of nn.SiLU()
    @staticmethod
    def execute(x):
        return x * jt.sigmoid(x)


class Hardswish(nn.Module):  # export-friendly version of nn.Hardswish()
    @staticmethod
    def execute(x):
        # return x * F.hardsigmoid(x)  # for torchscript and CoreML
        return x * nn.hardtanh(x + 3, 0., 6.) / 6.  # for torchscript, CoreML and ONNX


class MemoryEfficientSwish(nn.Module):
    class F(jt.Function):
        def execute(self, x):
            self.x = x
            return x * jt.sigmoid(x)

        def grad(self, grad_output):
            x = self.x
            sx = jt.sigmoid(x)
            return grad_output * (sx * (1 + x * (1 - sx)))

    def execute(self, x):
        return self.F()(x)


# Mish https://github.com/digantamisra98/Mish --------------------------------------------------------------------------
class Mish(nn.Module):
    @staticmethod
    def execute(x):
        return x * nn.softplus(x).tanh()


class MemoryEfficientMish(nn.Module):
    class F(jt.Function):
        def execute(self, x):
            self.x = x
            return x*(jt.tanh(nn.softplus(x)))  # x * tanh(ln(1 + exp(x)))

        def grad(self, grad_output):
            x = self.x
            sx = jt.sigmoid(x)
            fx = jt.softplus(x).tanh()
            return grad_output * (fx + x * sx * (1 - fx * fx))

    def execute(self, x):
        return self.F()(x)


# FReLU https://arxiv.org/abs/2007.11824 -------------------------------------------------------------------------------
class FReLU(nn.Module):
    def __init__(self, c1, k=3):  # ch_in, kernel
        super().__init__()
        self.conv = nn.Conv(c1, c1, k, 1, 1, groups=c1, bias=False)
        self.bn = nn.BatchNorm(c1)

    def execute(self, x):
        return jt.maximum(x, self.bn(self.conv(x)))
