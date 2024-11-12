import torch, math
from torch import nn


class AbstractPositionalEncoding(nn.Module):
    def create_pe_array(self,
                        encoding_dimension,
                        base_period,
                        initial_period,
                        max_len,
                        ):
        """
        precompute pe array and to avoid recomputation
        for default positional encoding, use
            base_period=10000^{2/additional_dim}
            initial_period=2pi
        Args:
            encoding_dimension: number of dimnesions to include in encoding
            base_period: multiply the period by this for each addional dim
            initial_period: start at this period
            max_len: handles sequences up to this long
        Returns: pe array shepe (1,max_len,additional_dim)
            pe[0,:,2i] and pe[0,:,2i+1] are sin (resp. cos) of
                (2pi * p/(initial_period*base_period^{2i}))
        """
        position = torch.arange(max_len).unsqueeze(1)

        # both of these are initial_period*base_period^{0,1,...},
        #   they are different lengths if additional_dim is odd
        period_term_even = initial_period*torch.pow(torch.tensor(base_period),
                                                    (torch.arange(0, encoding_dimension, 2))/2
                                                    )
        period_term_odd = initial_period*torch.pow(torch.tensor(base_period),
                                                   (torch.arange(1, encoding_dimension, 2) - 1)/2
                                                   )
        pe = torch.zeros(1, max_len, encoding_dimension)
        pe[0, :, 0::2] = torch.sin(2*torch.pi*position/period_term_even)
        pe[0, :, 1::2] = torch.cos(2*torch.pi*position/period_term_odd)
        return pe


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
    def __init__(self,
                 d_model: int,
                 dropout: float = 0.1,
                 base_period=None,
                 max_len: int = 5000,
                 ):
        super().__init__()
        if base_period is None:
            # this is arranged so that the period of the 2i and 2i+1 entries are 2pi*10000^(i/additional_dim)
            # for some reason the period is not an even number in the original positional encoder, so we can do that if we must
            base_period = math.pow(10000.0, 2/d_model)
            initial_period = (2*torch.pi)
        else:
            # the initial period being 1 is kind of useless, so start off at 1/base_period^1
            initial_period = base_period
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2)*(-math.log(10000.0)/d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position*div_term)
        pe[0, :, 1::2] = torch.cos(position*div_term)
        pe=self.create_pe_array(encoding_dimension=d_model,
                                base_period=base_period,
                                initial_period=initial_period,
                                max_len=max_len,
                                )

        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class PositionalAppender(AbstractPositionalEncoding):
    """
    appends positional encoding instead of doing the thing normal pos enc does
    this should prevent confusion at lower dimensional embedding sizes
    also has a linear layer that returns model to d_model dimension
    """

    def __init__(self,
                 d_model: int,
                 dropout: float = 0.1,
                 additional_dim=None,
                 base_period=None,
                 max_len: int = 5000,
                 ):
        """
        positional encoding at position p and dimension 2i is sin(2pi*p/(base_period^{2i})) 2i+1 is cos(that)
        i.e. if base_period is 2, the first two dimensions will encode sin or cos of (2pi * p / base period)
            note we shift by one to avoid the period being 1 and useless
        Args:
            d_model: dimension of model embedding
            dropout: dropout proportion
            additional_dim: additionall dimensions to append for positional encoding, defaults to d_model
            base_period: base period for positional encoding, defaults to 10000^{2/additional_dim}
            max_len: max seq length
        """
        super().__init__()
        if additional_dim is None:
            additional_dim = d_model
        self.additional_dim = additional_dim
        if base_period is None:
            # this is arranged so that the period of the 2i and 2i+1 entries are 2pi*10000^(i/additional_dim)
            # for some reason the period is not an even number in the original positional encoder, so we can do that if we must
            base_period = math.pow(10000.0, 2/self.additional_dim)
            initial_period = (2*torch.pi)
        else:
            # the initial period being 1 is kind of useless, so start off at 1/base_period^1
            initial_period = base_period
        self.dropout = nn.Dropout(p=dropout)
        pe = self.create_pe_array(encoding_dimension=additional_dim,
                                  base_period=base_period,
                                  initial_period=initial_period,
                                  max_len=max_len,
                                  )

        self.register_buffer('pe', pe)
        self.linear = nn.Linear(d_model + self.additional_dim, d_model)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        # truncat/expand pe to the shape (batch size, seq len, additional_dim)
        thing = self.pe[:, :x.size(1), :].expand(list(x.shape)[:-1] + [-1])
        appended = torch.cat((x, thing), dim=-1)
        return self.dropout(self.linear(appended))


if __name__ == '__main__':
    import numpy as np

    additional_dim = 5
    encoder = PositionalAppender(d_model=1,
                                 dropout=0,
                                 additional_dim=additional_dim,
                                 base_period=2,
                                 )
    x = torch.zeros(1, 5, 1)
    print(encoder.forward(x))
    print(encoder.pe[:, :x.size(1), :].expand(list(x.shape)[:-1] + [-1]))
    print(encoder.pe[:, :10])

    # default obtained from https://www.geeksforgeeks.org/positional-encoding-in-transformers/
    additional_dim = 10  # make this even
    position = x.size(1)
    angle_rads = np.arange(position)[:, np.newaxis]/np.power(10000,
                                                             (2*(np.arange(additional_dim)[np.newaxis,
                                                                 :]//2))/np.float32(
                                                                 additional_dim))

    # Apply sine to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # Apply cosine to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]
    pos_encoding = torch.tensor(pos_encoding)

    # check if the thingy is the same
    encoder = PositionalAppender(d_model=1,
                                 dropout=0,
                                 additional_dim=additional_dim,
                                 base_period=None,
                                 max_len=position,
                                 )
    print(encoder.pe)
    print(pos_encoding)
    print(encoder.pe - pos_encoding)
    assert torch.all(torch.isclose(encoder.pe, pos_encoding, ))
