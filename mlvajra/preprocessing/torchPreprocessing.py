import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Function
from numpy.fft import rfft2, irfft2


class BadFFTFunction(Function):

    def forward(self, input):
        numpy_input = input.detach().numpy()
        result = abs(rfft2(numpy_input))
        return input.new(result)

    def backward(self, grad_output):
        numpy_go = grad_output.numpy()
        result = irfft2(numpy_go)
        return grad_output.new(result)

# since this layer does not have any parameters, we can
# simply declare this as a function, rather than as an nn.Module class


def incorrect_fft(input):
    return BadFFTFunction()(input)
