import torch
from torch import nn
from torch.nn import init

from models.hyperlstm import HyperLSTMCell


class PTBModel(nn.Module):

    def __init__(self, num_chars, input_size, hidden_size,
                 hyper_hidden_size, hyper_embedding_size,
                 use_layer_norm, dropout_prob):
        super().__init__()
        self.char_embedding = nn.Embedding(num_embeddings=num_chars,
                                           embedding_dim=input_size)
        self.hyperlstm_cell = HyperLSTMCell(
            input_size=input_size, hidden_size=hidden_size,
            hyper_hidden_size=hyper_hidden_size,
            hyper_embedding_size=hyper_embedding_size,
            use_layer_norm=use_layer_norm, dropout_prob=dropout_prob)
        self.output_proj = nn.Linear(in_features=hidden_size,
                                     out_features=num_chars)
        self.dropout = nn.Dropout(dropout_prob)
        self.reset_parameters()

    def reset_parameters(self):
        init.normal(self.char_embedding.weight.data, mean=0, std=0.01)
        self.hyperlstm_cell.reset_parameters()
        init.xavier_uniform(self.output_proj.weight.data)

    def forward(self, inputs, state, hyper_state):
        inputs_emb = self.char_embedding(inputs)
        max_length = inputs.size(0)

        inputs_emb = self.dropout(inputs_emb)
        rnn_outputs = []
        for t in range(max_length):
            output, state, hyper_state = self.hyperlstm_cell(
                x=inputs_emb[t], state=state, hyper_state=hyper_state)
            rnn_outputs.append(output)
        rnn_outputs = torch.stack(rnn_outputs, dim=0)
        rnn_outputs = self.dropout(rnn_outputs)
        logits = self.output_proj(rnn_outputs)
        return logits, state, hyper_state
