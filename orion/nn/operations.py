import math
import torch

from .module import Module, timer
from .linear import Permutation

class Add(Module):
    def __init__(self):
        super().__init__()
        self.set_depth(0)

    def compute_fhe_output_shape(self, **kwargs):
        fhe_input_shape = kwargs["fhe_input_shape"]

        if isinstance(fhe_input_shape, list):
            return fhe_input_shape[0]
        return fhe_input_shape

    def forward(self, x, y):
        return x + y
    

class Mult(Module):
    def __init__(self):
        super().__init__()
        self.set_depth(1)

    def compute_fhe_output_shape(self, **kwargs):
        fhe_input_shape = kwargs["fhe_input_shape"]

        if isinstance(fhe_input_shape, list):
            return fhe_input_shape[0]
        return fhe_input_shape

    def forward(self, x, y):
        return x * y
    
# Summation along the given dimension. Current implementation is already
# robust, nevertheless, it still has the following constraints on the input:
# 1. The input gap is 1; 2. either there is only 1 input ciphertext, or the
# dimensions have been padded to PO2.
class LogSum(Module):
    def __init__(self, dim):
        super().__init__()
        self.set_depth(0)
        self.dim = dim
        self.triple = [0, 0, 0]
        self.in_stride, self.out_stride, self.group = 0, 0, 0
        self.slots = 0

    def prepare_params(self, shape):
        self.slots = self.scheme.params.get_slots()
        self.triple[0] = math.prod(shape[self.dim+1:])
        self.triple[1] = shape[self.dim]
        self.triple[2] = math.prod(shape[:self.dim])
        
        if self.triple[0] >= self.slots:
            self.in_stride, self.out_stride, self.group = \
                self.slots, math.ceil(self.triple[0] / self.slots), self.triple[2]
        elif self.triple[0] * self.triple[1] >= self.slots or self.triple[2] == 1:
            self.in_stride, self.out_stride, self.group = \
                self.triple[0], 1, self.triple[2]
        else:
            raise(ValueError, "The layout of the ciphertext does not allow \
                trivial summation. Please call permutation first. You can \
                always move dim to 0 to solve this problem.")
        
        return self.in_stride, self.out_stride, self.group

    def compute_fhe_output_shape(self, **kwargs):
        fhe_input_shape = kwargs["fhe_input_shape"]
        self.prepare_params(list(fhe_input_shape))

        if isinstance(fhe_input_shape, list):
            return fhe_input_shape[0]
        return fhe_input_shape

    def forward(self, x):
        if self.he_mode:
            x_sum = x.summation(self.out_stride, self.group)
            s = self.in_stride
            while s < self.slots:
                x_sum += x_sum.roll(s, in_place=False)
                s *= 2
            x_sum = x_sum.broadcast(len(x) // len(x_sum))
            return x_sum
        else:
            return torch.sum(x, self.dim)


# With permutation implemented, we can concatenate along any dim.
# However, as the number of input tensors is not specified at compile
# time, we can only concat tensors of the same shape when dim != 0.
# For more general cases, users should insert permutation layers before
# concatenation to move the concat dim to front.
class Cat(Module):
    def __init__(self):
        super().__init__()
        self.set_depth(0)

    def compute_fhe_output_shape(self, **kwargs):
        input_shape = kwargs["input_shape"]
        clear_output_shape = kwargs["clear_output_shape"]
        fhe_input_shape = kwargs["fhe_input_shape"]
        
        for i in range(len(input_shape[0])):
            if input_shape[0][i] != clear_output_shape[i]: dim = i; break
        fhe_output_shape = list(fhe_input_shape[0])
        for shape in fhe_input_shape[1:]:
            fhe_output_shape[dim] += shape[dim]

        return torch.Size(fhe_output_shape)
    
    def forward(self, x_list, dim=0):
        if self.he_mode:
            out = x_list[0]
            for x in x_list[1:]:
                out.cat(x, dim=dim)
            return out
        else: return torch.cat(x_list, dim=dim)


class Bootstrap(Module):
    def __init__(self, input_min, input_max, input_level):
        super().__init__()
        self.input_min = input_min 
        self.input_max = input_max 
        self.input_level = input_level
        self.prescale = 1
        self.postscale = 1
        self.constant = 0

    def extra_repr(self):
        l_eff = len(self.scheme.params.get_logq()) - 1
        return f"l_eff={l_eff}"

    def fit(self):
        center = (self.input_min + self.input_max) / 2 
        half_range = (self.input_max - self.input_min) / 2
        self.low = (center - (self.margin * half_range))
        self.high = (center + (self.margin * half_range))

        # We'll want to scale from [A, B] into [-1, 1] using a value of the
        # form 1 / integer, so that way our multiplication back to the range
        # [A, B] (by integer) after bootstrapping doesn't consume a level.
        if self.high - self.low > 2:
            self.postscale = math.ceil((self.high - self.low) / 2)
            self.prescale = 1 / self.postscale

        self.constant = -(self.low + self.high) / 2 

    def compile(self):
        # We'll then encode the prescale at the level of the input ciphertext
        # to ensure its rescaling is errorless
        elements = self.fhe_input_shape.numel()
        curr_slots = 2 ** math.ceil(math.log2(elements))

        prescale_vec = torch.zeros(curr_slots)
        prescale_vec[:elements] = self.prescale

        ql = self.scheme.encoder.get_moduli_chain()[self.input_level]
        self.prescale_ptxt = self.scheme.encoder.encode(
            prescale_vec, level=self.input_level, scale=ql)

    @timer
    def forward(self, x):
        if not self.he_mode:
            return x
        
        # Shift and scale into range [-1, 1]. Important caveat -- here we first
        # shift, then scale. This let's us zero out unused slots and enables
        # sparse bootstrapping (i.e., where slots < N/2).
        if self.constant != 0:
            x += self.constant
        x *= self.prescale_ptxt
 
        x = x.bootstrap()

        # Scale and shift back to the original range
        if self.postscale != 1:
            x *= self.postscale 
        if self.constant != 0:
            x -= self.constant

        return x




