import sys
import math
import copy
import torch

class PlainTensor:
    def __init__(self, scheme, ptxt_ids, shape, on_shape=None):
        self.scheme = scheme
        self.backend = scheme.backend
        self.encoder = scheme.encoder
        
        self.ids = [ptxt_ids] if isinstance(ptxt_ids, int) else ptxt_ids
        self.shape = shape 
        self.on_shape = on_shape or shape

    def __del__(self):
        if 'sys' in globals() and sys.modules and self.scheme:
            try:
                for idx in self.ids:
                    self.backend.DeletePlaintext(idx)
            except Exception: 
                pass # avoids errors for GC at program termination

    def __len__(self):
        return len(self.ids)
    
    def __str__(self):
        return str(self.decode())
    
    def mul(self, other, in_place=False):
        if not isinstance(other, CipherTensor):
            raise ValueError(f"Multiplication between PlainTensor and "
                             f"{type(other)} is not supported.")

        mul_ids = []
        for i in range(len(self.ids)):
            mul_id = self.evaluator.mul_ciphertext(
                other.ids[i], self.ids[i], in_place)
            mul_ids.append(mul_id)

        if in_place:
            return other
        return CipherTensor(self.scheme, mul_ids, self.shape, self.on_shape) 

    def __mul__(self, other):
        return self.mul(other, in_place=False)     

    def __imul__(self, other):
        return self.mul(other, in_place=True)
    
    def _check_valid(self, other):
        return 
        
    def get_ids(self):
        return self.ids
    
    def scale(self):
        return self.backend.GetPlaintextScale(self.ids[0])
    
    def set_scale(self, scale):
        for ptxt in self.ids:
            self.backend.SetPlaintextScale(ptxt, scale)

    def level(self):
        return self.backend.GetPlaintextLevel(self.ids[0])
    
    def slots(self):
        return self.backend.GetPlaintextSlots(self.ids[0])
    
    def min(self):
        return self.decode().min()
    
    def max(self):
        return self.decode().max()
    
    def moduli(self):
        return self.backend.GetModuliChain()
    
    def decode(self):
        return self.encoder.decode(self)
    

class CipherTensor:
    def __init__(self, scheme, ctxt_ids, shape, on_shape=None):
        self.scheme = scheme
        self.backend = scheme.backend 
        self.encryptor = scheme.encryptor
        self.evaluator = scheme.evaluator
        self.bootstrapper = scheme.bootstrapper

        self.ids = [ctxt_ids] if isinstance(ctxt_ids, int) else ctxt_ids 
        self.shape = shape 
        self.on_shape = on_shape or shape

    def __del__(self):
        if 'sys' in globals() and sys.modules and self.scheme:
            try:
                for idx in self.ids:
                    self.backend.DeleteCiphertext(idx)
            except Exception: 
                pass # avoids errors for GC at program termination

    def __len__(self):
        return len(self.ids)
    
    def __str__(self):
        ptxt = self.decrypt()
        return str(ptxt.decode())
    
    #--------------#
    #  Operations  #
    #--------------#
    
    def __neg__(self):
        neg_ids = []
        for ctxt in self.ids:
            neg_id = self.evaluator.negate(ctxt)
            neg_ids.append(neg_id)

        return CipherTensor(self.scheme, neg_ids, self.shape, self.on_shape)
    
    def add(self, other, in_place=False):
        self._check_valid(other)

        add_ids = []
        for i in range(len(self.ids)):
            if isinstance(other, (int, float)):
                add_id = self.evaluator.add_scalar(
                    self.ids[i], other, in_place)
            elif isinstance(other, PlainTensor):
                add_id = self.evaluator.add_plaintext(
                    self.ids[i], other.ids[i], in_place)
            elif isinstance(other, CipherTensor):
                add_id = self.evaluator.add_ciphertext(
                    self.ids[i], other.ids[i], in_place)
            else:
                raise ValueError(f"Addition between CipherTensor and "
                                 f"{type(other)} is not supported.")

            add_ids.append(add_id)

        if in_place:
            return self
        return CipherTensor(self.scheme, add_ids, self.shape, self.on_shape)
    
    def __add__(self, other):
        return self.add(other, in_place=False)
    
    def __radd__(self, other):
        return self.add(other, in_place=False)

    def __iadd__(self, other):
        return self.add(other, in_place=True)
    
    def sub(self, other, in_place=False):
        self._check_valid(other)

        sub_ids = []
        for i in range(len(self.ids)):
            if isinstance(other, (int, float)):
                sub_id = self.evaluator.sub_scalar(
                    self.ids[i], other, in_place)
            elif isinstance(other, PlainTensor):
                sub_id = self.evaluator.sub_plaintext(
                    self.ids[i], other.ids[i], in_place)
            elif isinstance(other, CipherTensor):
                sub_id = self.evaluator.sub_ciphertext(
                    self.ids[i], other.ids[i], in_place)
            else:
                raise ValueError(f"Subtraction between CipherTensor and "
                                 f"{type(other)} is not supported.")

            sub_ids.append(sub_id)

        if in_place:
            return self
        return CipherTensor(self.scheme, sub_ids, self.shape, self.on_shape)
    
    def __sub__(self, other):
        return self.sub(other, in_place=False)

    def __isub__(self, other):
        return self.sub(other, in_place=True)

    def __rsub__(self, other):
        return self.sub(other, in_place=False)
    
    def mul(self, other, in_place=False):
        self._check_valid(other)

        mul_ids = []
        for i in range(len(self.ids)):
            if isinstance(other, (int, float)):
                mul_id = self.evaluator.mul_scalar(
                    self.ids[i], other, in_place)
            elif isinstance(other, PlainTensor):
                mul_id = self.evaluator.mul_plaintext(
                    self.ids[i], other.ids[i], in_place)
            elif isinstance(other, CipherTensor):
                mul_id = self.evaluator.mul_ciphertext(
                    self.ids[i], other.ids[i], in_place)
            else:
                raise ValueError(f"Multiplication between CipherTensor and "
                                 f"{type(other)} is not supported.")
            
            mul_ids.append(mul_id)

        if in_place:
            return self
        return CipherTensor(self.scheme, mul_ids, self.shape, self.on_shape) 
    
    def __mul__(self, other):
        return self.mul(other, in_place=False)     

    def __imul__(self, other):
        return self.mul(other, in_place=True)

    def __rmul__(self, other):
        return self.mul(other, in_place=False)
    
    def roll(self, amount, in_place=False):
        rot_ids = []
        for ctxt in self.ids:
            rot_id = self.evaluator.rotate(ctxt, amount, in_place)
            rot_ids.append(rot_id)

        return CipherTensor(self.scheme, rot_ids, self.shape, self.on_shape)

    # This function is expected to be used in pair with broadcast, so it does
    # not change shape or on_shape for simplicity.
    def summation(self, stride=1, group=1):
        total_len = len(self.ids)
        if total_len % group != 0:
            raise ValueError(f"Input length ({total_len}) must be divisible by group ({group})")
            
        len_per_group = total_len // group
        sum_ids = [0] * (stride * group) 
        
        for g in range(group):
            input_offset = g * len_per_group
            output_offset = g * stride
            for i in range(len_per_group):
                real_input_idx = input_offset + i
                real_output_idx = output_offset + (i % stride)
                if i < stride:
                    sum_ids[real_output_idx] = self.ids[real_input_idx]
                else:
                    sum_ids[real_output_idx] = self.evaluator.add_ciphertext(
                        sum_ids[real_output_idx], self.ids[real_input_idx], False
                    )
        
        return CipherTensor(self.scheme, sum_ids, self.shape, self.on_shape)

    def broadcast(self, times=1):
        ids = copy.deepcopy(self.ids)
        for _ in range(times - 1):
            for id in ids:
                self.ids.append(self.backend.CloneCiphertext(id))
        return CipherTensor(self.scheme, self.ids, self.shape, self.on_shape)
    
    def cat(self, other, dim=1):
        if isinstance(other, CipherTensor):
            for i, (shape1, shape2) in enumerate(
                zip(self.on_shape, other.on_shape)):
                if i == dim: continue
                if shape1 != shape2:
                    raise ValueError(f"Dimension {i} mismatch: "
                                     f"({shape1}, {shape2}).")
            if self.shape.numel() + other.shape.numel() > self.slots():
                for id in other.ids:
                    self.ids.append(self.backend.CloneCiphertext(id))
            else:
                self += other.roll(self.slots() - self.shape.numel(), in_place=True)
            self.shape, self.on_shape = list(self.shape), list(self.on_shape)
            self.shape[dim] += other.shape[dim]
            self.on_shape[dim] += other.on_shape[dim]
            self.shape, self.on_shape = (
                torch.Size(self.shape), torch.Size(self.on_shape))
        else:
            raise TypeError("Concatenation is only supported between"
                            "ciphertensors.")
        return
    
    def _check_valid(self, other):
        return
    
    #----------------------
    #
    #---------------------
    
    def scale(self):
        return self.backend.GetCiphertextScale(self.ids[0])
    
    def set_scale(self, scale):
        for ctxt in self.ids:
            self.backend.SetCiphertextScale(ctxt, scale)

    def level(self):
        return self.backend.GetCiphertextLevel(self.ids[0])
    
    def slots(self):
        return self.backend.GetCiphertextSlots(self.ids[0])
    
    def degree(self):
        return self.backend.GetCiphertextDegree(self.ids[0])
    
    def min(self):
        return self.decrypt().min()
    
    def max(self):
        return self.decrypt().max()
    
    def moduli(self):
        return self.backend.GetModuliChain()
    
    def bootstrap(self):
        elements = self.on_shape.numel()
        slots = 2 ** math.ceil(math.log2(elements))
        slots = int(min(self.slots(), slots)) # sparse bootstrapping
        
        btp_ids = []
        for ctxt in self.ids:
            btp_id = self.bootstrapper.bootstrap(ctxt, slots)
            btp_ids.append(btp_id)

        return CipherTensor(self.scheme, btp_ids, self.shape, self.on_shape)
        
    def decrypt(self):
        return self.encryptor.decrypt(self)


def new_plaintext(value, scheme, level=None, scale=None):
    if isinstance(value, (int, float, complex)):
        slots = scheme.params.get_slots()
        vector = [value] * slots
    elif isinstance(value, list):
        vector = value
    else:
        raise(TypeError, "Value should be list, int, float or complex type.")
    
    return scheme.encode(vector, level=None, scale=None)

def new_ciphertext(value, scheme, level=None, scale=None):
    plaintext = new_plaintext(value, scheme, level, scale)
    return scheme.encrypt(plaintext)
