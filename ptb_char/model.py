import torch
from torch import nn
from torch.nn import init

from models.hyperlstm import HyperLSTMCell, LSTMCell


class PTBModel(nn.Module):

    def __init__(self, rnn_type, num_chars, input_size, hidden_size,
                 hyper_hidden_size, hyper_embedding_size,
                 use_layer_norm, dropout_prob):
        super().__init__()
        self.rnn_type = rnn_type
        self.num_chars = num_chars
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.hyper_hidden_size = hyper_hidden_size
        self.hyper_embedding_size = hyper_embedding_size
        self.use_layer_norm = use_layer_norm
        self.dropout_prob = dropout_prob

        self.char_embedding = nn.Embedding(num_embeddings=num_chars,
                                           embedding_dim=input_size)
        if rnn_type == 'hyperlstm':
            self.rnn_cell = HyperLSTMCell(
                input_size=input_size, hidden_size=hidden_size,
                hyper_hidden_size=hyper_hidden_size,
                hyper_embedding_size=hyper_embedding_size,
                use_layer_norm=use_layer_norm, dropout_prob=dropout_prob)
        elif rnn_type == 'lstm':
            self.rnn_cell = LSTMCell(
                input_size=input_size, hidden_size=hidden_size,
                use_layer_norm=use_layer_norm, dropout_prob=dropout_prob)
        else:
            raise ValueError('Unknown RNN type')
        self.output_proj = nn.Linear(in_features=hidden_size,
                                     out_features=num_chars)
        self.dropout = nn.Dropout(dropout_prob)
        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform(self.char_embedding.weight.data)
        self.rnn_cell.reset_parameters()
        init.xavier_uniform(self.output_proj.weight.data)
        init.constant(self.output_proj.bias.data, val=0)

    def forward(self, inputs, state, hyper_state=None):
        inputs_emb = self.char_embedding(inputs)
        max_length = inputs.size(0)

        inputs_emb = self.dropout(inputs_emb)
        rnn_outputs = []
        for t in range(max_length):
            if self.rnn_type == 'hyperlstm':
                output, state, hyper_state = self.rnn_cell(
                    x=inputs_emb[t], state=state, hyper_state=hyper_state)
            elif self.rnn_type == 'lstm':
                output, state = self.rnn_cell.forward(
                    x=inputs_emb[t], state=state)
            else:
                raise ValueError('Unknown RNN type')
            rnn_outputs.append(output)
        rnn_outputs = torch.stack(rnn_outputs, dim=0)
        rnn_outputs = self.dropout(rnn_outputs)
        logits = self.output_proj(rnn_outputs)
        return logits, state, hyper_state
