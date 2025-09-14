import torch
import warnings
import torch.nn as nn
import orion.nn as on
from orion.backend.python.tensors import CipherTensor


class RNNUnit(on.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()

        self.hidden_size = hidden_size
        self.input_linears = nn.ModuleList()
        self.hidden_linears = nn.ModuleList()
        self.act = nn.ModuleList()

        self.act.append(on.Tanh())
        self.input_linears.append(on.Linear(input_size, hidden_size))
        self.hidden_linears.append(on.Linear(hidden_size, hidden_size, bias=True))
        for _ in range(1, num_layers):
            self.act.append(on.Tanh())
            self.input_linears.append(on.Linear(hidden_size, hidden_size))
            self.hidden_linears.append(on.Linear(hidden_size, hidden_size, bias=True))

    def forward(self, x, hidden_state, first_block=False):
        out = []

        if first_block:
            new_state = self.act[0](self.input_linears[0](x))
            out.append(new_state)
            for i in range(1, len(self.input_linears)):
                new_state = self.act[i](self.input_linears[i](new_state))
                out.append(new_state)
        else:
            if len(hidden_state) != len(self.input_linears):
                raise ValueError("Length of hidden_state must match num_layers.")
            new_state = self.act[0](self.input_linears[0](x) 
                                 + self.hidden_linears[0](hidden_state[0]))
            out.append(new_state)
            for i in range(1, len(self.input_linears)):
                new_state = self.act[i](
                    self.input_linears[i](new_state) 
                    + self.hidden_linears[i](hidden_state[i])
                    )
                out.append(new_state)
        return out


class RNNModel(on.Module):
    def __init__(self, input_size, hidden_size, max_length, num_layers):
        super().__init__()
        self.rnn = nn.ModuleList()
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.hidden_state = None
        for _ in range(max_length):
            self.rnn.append(RNNUnit(input_size, hidden_size, num_layers))

    def forward(self, x): 
        result = []
        out = self.rnn[0](x[0], None, first_block=True)
        self.hidden_state = out
        result.append(out[-1])
        for i in range(1, self.max_length):
            out = self.rnn[i](x[i], self.hidden_state)
            self.hidden_state = out
            result.append(out[-1])
        return result
    

class RNN():
    def __init__(self, input_size, hidden_size, max_length, num_layers=1, batch_first=False):
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1.")
        if num_layers > 1:
            warnings.warn("RNN currently only supports num_layers=1. Ignoring the provided value.")
            num_layers = 1
        self.batch_first = batch_first
        self.model = RNNModel(input_size, hidden_size, max_length, num_layers)

    def __call__(self, x):
        if isinstance(x, torch.Tensor):
            if self.model.he_mode:
                raise ValueError("Input must be a CipherTensor in FHE mode.")
            if len(x.shape) != 3:
                raise ValueError("Input tensor must be 3-dimensional.")
            if self.batch_first:
                x = x.permute(1, 0, 2)
            if x.size(0) > self.model.max_length:
                warnings.warn(f"Input sequence length {x.size(0)} exceeds max_length."
                                + f"Truncating to {self.model.max_length}.")
            out = self.model(x)
            out = torch.stack(out, dim=0)
            if self.batch_first:
                out = out.permute(1, 0, 2)
        elif isinstance(x, CipherTensor):
            if not self.model.he_mode:
                raise ValueError("Input must be a torch.Tensor in cleartext mode.")
            if len(x) > self.model.max_length:
                warnings.warn(f"Input sequence length {len(x)} exceeds max_length."
                                + f"Truncating to {self.model.max_length}.")
            out = self.model(x)
            out = [o.decrypt().decode() for o in out]
            out = torch.stack(out, dim=0)
            if self.batch_first:
                out = out.permute(1, 0, 2)
        
        return out
    
    def he(self):
        self.model.he()

    def eval(self):
        self.model.eval()
