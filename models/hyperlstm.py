import torch
from torch import nn
from torch.autograd.variable import Variable
from torch.nn import init


class LayerNorm(nn.Module):

    """
    Implementation of layer normalization, slightly modified from
    https://github.com/pytorch/pytorch/issues/1959.
    """

    def __init__(self, num_features, eps=1e-6):
        super().__init__()
        self.num_features = num_features
        self.eps = eps

        self.gamma = nn.Parameter(torch.FloatTensor(num_features))
        self.beta = nn.Parameter(torch.FloatTensor(num_features))
        self.reset_parameters()

    def reset_parameters(self):
        init.constant(self.gamma.data, val=1)
        init.constant(self.beta.data, val=0)

    def forward(self, input):
        mean = input.mean(dim=-1, keepdim=True)
        std = input.std(dim=-1, keepdim=True)
        return self.gamma*(input - mean)/(std + self.eps) + self.beta


class ParallelLayerNorm(nn.Module):

    """
    Faster parallel layer normalization.
    Inspired by the implementation of
    https://github.com/hardmaru/supercell/blob/master/supercell.py.
    """

    def __init__(self, num_inputs, num_features, eps=1e-6):
        super().__init__()
        self.num_inputs = num_inputs
        self.num_features = num_features
        self.eps = eps

        self.gamma = nn.Parameter(torch.FloatTensor(num_inputs, num_features))
        self.beta = nn.Parameter(torch.FloatTensor(num_inputs, num_features))
        self.reset_parameters()

    def reset_parameters(self):
        init.constant(self.gamma.data, val=1)
        init.constant(self.beta.data, val=0)

    def forward(self, *inputs):
        """
        Args:
            input_1, ... (Variable): Variables to which
                layer normalization be applied. The number of inputs
                must be identical to self.num_inputs.
        """

        inputs_stacked = torch.stack(inputs, dim=-2)
        mean = inputs_stacked.mean(dim=-1, keepdim=True)
        std = inputs_stacked.std(dim=-1, keepdim=True)
        outputs_stacked = (self.gamma*(inputs_stacked - mean)/(std + self.eps)
                           + self.beta)
        outputs = torch.unbind(outputs_stacked, dim=-2)
        return outputs


class LSTMCell(nn.Module):

    def __init__(self, input_size, hidden_size, use_layer_norm,
                 dropout_prob=0):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_layer_norm = use_layer_norm
        self.dropout_prob = dropout_prob

        self.linear_ih = nn.Linear(in_features=input_size,
                                   out_features=4 * hidden_size)
        self.linear_hh = nn.Linear(in_features=hidden_size,
                                   out_features=4 * hidden_size,
                                   bias=False)
        self.dropout = nn.Dropout(dropout_prob)
        if use_layer_norm:
            self.ln_ifgo = ParallelLayerNorm(num_inputs=4,
                                             num_features=hidden_size)
            self.ln_c = LayerNorm(hidden_size)
        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform(self.linear_ih.weight.data)
        init.constant(self.linear_ih.bias.data, val=0)
        init.orthogonal(self.linear_hh.weight.data)
        if self.use_layer_norm:
            self.ln_ifgo.reset_parameters()
            self.ln_c.reset_parameters()

    def forward(self, x, state):
        if state is None:
            batch_size = x.size(0)
            zero_state = Variable(
                x.data.new(batch_size, self.hidden_size).zero_())
            state = (zero_state, zero_state)
        h, c = state
        lstm_vector = self.linear_ih(x) + self.linear_hh(h)
        i, f, g, o = lstm_vector.chunk(chunks=4, dim=1)
        if self.use_layer_norm:
            i, f, g, o = self.ln_ifgo(i, f, g, o)
        f = f + 1
        new_c = c*f.sigmoid() + i.sigmoid()*self.dropout(g.tanh())
        if self.use_layer_norm:
            new_c = self.ln_c(new_c)
        new_h = new_c.tanh() * o.sigmoid()
        new_state = (new_h, new_c)
        return new_h, new_state


class HyperLSTMCell(nn.Module):

    def __init__(self, input_size, hidden_size,
                 hyper_hidden_size, hyper_embedding_size,
                 use_layer_norm, dropout_prob):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.hyper_hidden_size = hyper_hidden_size
        self.hyper_embedding_size = hyper_embedding_size
        self.use_layer_norm = use_layer_norm
        self.dropout_prob = dropout_prob

        self.hyper_cell = LSTMCell(input_size=input_size + hidden_size,
                                   hidden_size=hyper_hidden_size,
                                   use_layer_norm=use_layer_norm)
        # Hyper LSTM: Projection
        for y in ('i', 'f', 'g', 'o'):
            proj_h = nn.Linear(in_features=hyper_hidden_size,
                               out_features=hyper_embedding_size)
            proj_x = nn.Linear(in_features=hyper_hidden_size,
                               out_features=hyper_embedding_size)
            proj_b = nn.Linear(in_features=hyper_hidden_size,
                               out_features=hyper_embedding_size,
                               bias=False)
            setattr(self, f'hyper_proj_{y}h', proj_h)
            setattr(self, f'hyper_proj_{y}x', proj_x)
            setattr(self, f'hyper_proj_{y}b', proj_b)
        # Hyper LSTM: Scaling
        for y in ('i', 'f', 'g', 'o'):
            scale_h = nn.Linear(in_features=hyper_embedding_size,
                                out_features=hidden_size,
                                bias=False)
            scale_x = nn.Linear(in_features=hyper_embedding_size,
                                out_features=hidden_size,
                                bias=False)
            scale_b = nn.Linear(in_features=hyper_embedding_size,
                                out_features=hidden_size,
                                bias=False)
            setattr(self, f'hyper_scale_{y}h', scale_h)
            setattr(self, f'hyper_scale_{y}x', scale_x)
            setattr(self, f'hyper_scale_{y}b', scale_b)
        self.linear_ih = nn.Linear(in_features=input_size,
                                   out_features=4 * hidden_size,
                                   bias=False)
        self.linear_hh = nn.Linear(in_features=hidden_size,
                                   out_features=4 * hidden_size,
                                   bias=False)
        self.bias = nn.Parameter(torch.FloatTensor(4 * hidden_size))
        if use_layer_norm:
            self.ln_ifgo = ParallelLayerNorm(num_inputs=4,
                                             num_features=hidden_size)
            self.ln_c = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.reset_parameters()

    def reset_parameters(self):
        # Hyper LSTM
        self.hyper_cell.reset_parameters()
        # Hyper LSTM: Projection
        for y in ('i', 'g', 'f', 'o'):
            proj_h = getattr(self, f'hyper_proj_{y}h')
            proj_x = getattr(self, f'hyper_proj_{y}x')
            proj_b = getattr(self, f'hyper_proj_{y}b')
            init.constant(proj_h.weight.data, val=0)
            init.constant(proj_h.bias.data, val=1)
            init.constant(proj_x.weight.data, val=0)
            init.constant(proj_x.bias.data, val=1)
            init.normal(proj_b.weight.data, mean=0, std=0.01)
        # Hyper LSTM: Scaling
        for y in ('i', 'g', 'f', 'o'):
            scale_h = getattr(self, f'hyper_scale_{y}h')
            scale_x = getattr(self, f'hyper_scale_{y}x')
            scale_b = getattr(self, f'hyper_scale_{y}b')
            init.constant(scale_h.weight.data,
                          val=0.1 / self.hyper_embedding_size)
            init.constant(scale_x.weight.data,
                          val=0.1 / self.hyper_embedding_size)
            init.constant(scale_b.weight.data, val=0)

        # Main LSTM
        init.xavier_uniform(self.linear_ih.weight.data)
        init.orthogonal(self.linear_hh.weight.data)
        init.constant(self.bias.data, val=0)

        # LayerNorm
        if self.use_layer_norm:
            self.ln_ifgo.reset_parameters()
            self.ln_c.reset_parameters()

    def compute_hyper_vector(self, hyper_h, name):
        proj = getattr(self, f'hyper_proj_{name}')
        scale = getattr(self, f'hyper_scale_{name}')
        return scale(proj(hyper_h))

    def forward(self, x, state, hyper_state, mask=None):
        """
        Args:
            x (Variable): A variable containing a float tensor
                of size (batch_size, input_size).
            state (tuple[Variable]): A tuple (h, c), each of which
                is of size (batch_size, hidden_size).
            hyper_state (tuple[Variable]): A tuple (hyper_h, hyper_c),
                each of which is of size (batch_size, hyper_hidden_size).
            mask (Variable): A variable containing a float tensor
                of size (batch_size,).

        Returns:
            state (tuple[Variable]): The current state of the main LSTM.
            hyper_state (tuple[Variable]): The current state of the
                hyper LSTM.
        """

        if state is None:
            batch_size = x.size(0)
            zero_state = Variable(
                x.data.new(batch_size, self.hidden_size).zero_())
            state = (zero_state, zero_state)

        h, c = state

        # Run a single step of Hyper LSTM.
        hyper_input = torch.cat([x, h], dim=1)
        new_hyper_h, new_hyper_state = self.hyper_cell(
            x=hyper_input, state=hyper_state)

        # Then, compute values for the main LSTM.
        xh = self.linear_ih(x)
        hh = self.linear_hh(h)

        ix, fx, gx, ox = xh.chunk(chunks=4, dim=1)
        ix = ix * self.compute_hyper_vector(hyper_h=new_hyper_h, name='ix')
        fx = fx * self.compute_hyper_vector(hyper_h=new_hyper_h, name='fx')
        gx = gx * self.compute_hyper_vector(hyper_h=new_hyper_h, name='gx')
        ox = ox * self.compute_hyper_vector(hyper_h=new_hyper_h, name='ox')

        ih, fh, gh, oh = hh.chunk(chunks=4, dim=1)
        ih = ih * self.compute_hyper_vector(hyper_h=new_hyper_h, name='ih')
        fh = fh * self.compute_hyper_vector(hyper_h=new_hyper_h, name='fh')
        gh = gh * self.compute_hyper_vector(hyper_h=new_hyper_h, name='gh')
        oh = oh * self.compute_hyper_vector(hyper_h=new_hyper_h, name='oh')

        ib, fb, gb, ob = self.bias.chunk(chunks=4, dim=0)
        ib = ib + self.compute_hyper_vector(hyper_h=new_hyper_h, name='ib')
        fb = fb + self.compute_hyper_vector(hyper_h=new_hyper_h, name='fb')
        gb = gb + self.compute_hyper_vector(hyper_h=new_hyper_h, name='gb')
        ob = ob + self.compute_hyper_vector(hyper_h=new_hyper_h, name='ob')

        i = ix + ih + ib
        f = fx + fh + fb + 1  # Set the initial forget bias to 1.
        g = gx + gh + gb
        o = ox + oh + ob

        if self.use_layer_norm:
            i, f, g, o = self.ln_ifgo(i, f, g, o)
        new_c = c*f.sigmoid() + self.dropout(g.tanh())*i.sigmoid()
        if self.use_layer_norm:
            new_c = self.ln_c(new_c)
        new_h = new_c.tanh() * o.sigmoid()

        # Apply the mask vector.
        if mask is not None:
            mask = mask.unsqueeze(1)
            new_h = new_h*mask + h*(1 - mask)
            new_c = new_c*mask + c*(1 - mask)

        new_state = (new_h, new_c)
        return new_h, new_state, new_hyper_state
