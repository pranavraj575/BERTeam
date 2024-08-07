import torch, math
from torch import nn


class AbstractPositionalEncoding(nn.Module):
    pass


class IdentityEncoding(AbstractPositionalEncoding):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        return x


class ClassicPositionalEncoding(AbstractPositionalEncoding):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2)*(-math.log(10000.0)/d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position*div_term)
        pe[0, :, 1::2] = torch.cos(position*div_term)

        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class PositionalAppender(AbstractPositionalEncoding):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2)*(-math.log(10000.0)/d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position*div_term)
        pe[0, :, 1::2] = torch.cos(position*div_term)

        self.register_buffer('pe', pe)
        self.linear = nn.Linear(2*d_model, d_model)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """

        appended = torch.cat((x, self.pe[:, :x.size(1), :].broadcast_to(x.shape)), dim=-1)
        return self.dropout(self.linear(appended))
