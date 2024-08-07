import torch
from torch import nn

from BERTeam.networks.positional_encoder import IdentityEncoding, ClassicPositionalEncoding, PositionalAppender


class InputEmbedder(nn.Module):
    def forward(self, pre_embedding, preembed_mask):
        """
        Args:
            pre_embedding: (N, S, *) observation preembedding
            mask: (N, S) boolean mask, or None
        Returns:
            (embedding (N, S', E), mask (N, S'))
        """
        raise NotImplementedError


class DiscreteInputEmbedder(InputEmbedder):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.embed = nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
        )
        self.pos_enc = IdentityEncoding()

    def forward(self, pre_embedding, preembed_mask):
        if pre_embedding is None:
            return None, None
        return self.pos_enc(self.embed(pre_embedding)), preembed_mask


class DiscreteInputPosEmbedder(DiscreteInputEmbedder):
    def __init__(self, num_embeddings, embedding_dim, dropout=.1):
        super().__init__(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        self.embed = nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
        )
        self.pos_enc = ClassicPositionalEncoding(d_model=embedding_dim, dropout=dropout)


class DiscreteInputPosAppender(DiscreteInputEmbedder):
    """
    instead of adding positional encoding, just appends it, and also has a linear layer to match dimensions
    """

    def __init__(self, num_embeddings, embedding_dim, dropout=.1):
        super().__init__(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        self.embed = nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
        )
        self.pos_enc = PositionalAppender(d_model=embedding_dim, dropout=dropout)


class LSTEmbedding(InputEmbedder):
    def __init__(self,
                 input_dim,
                 embedding_dim=512,
                 layers=4,
                 dropout=0,
                 device=None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        # skip batch and sequence dims
        self.flat = nn.Flatten(start_dim=2, end_dim=-1)
        self.lstm = nn.LSTM(input_size=input_dim,
                            hidden_size=embedding_dim,
                            num_layers=layers,
                            batch_first=True,
                            dropout=dropout,
                            bidirectional=False,
                            device=device,
                            )
        self.device = device
        self.embedding_dim = embedding_dim

    def subseqs_from_mask(self, mask):
        """
        splits a sequence according to where the mask is true
        Args:
            seq: (S,*) sequence
            mask: (S,) boolean mask of truth values
        Returns:
            iterable of (i,j) start end indices of each contiguous subsequence
        """
        out = []
        i = 0
        j = 0
        for j, value in enumerate(mask):
            if value:
                # this index is masked, so add the previous contiguous subsequence, if nonempty
                if i != j:
                    yield (i, j)

                # at best, next contiguous subsequence can start at next index
                i = j + 1
        # j is now len(mask)-1
        if i != j + 1:
            # if we end with an unmasked value, add the suffix in
            yield (i, j + 1)
        return out

    def output_from_lstm(self, seq, subseq=None):
        """
        gets output from a sequence
        Args:
            seq: (S,input_dim) sequence
            subseq: i,j start end
                uses 0, len(seq) by default
        Returns: (embed_dim,)
        """
        if subseq is not None:
            i, j = subseq
            seq = seq[i:j]
        out, _ = self.lstm.forward(seq)

        return out[-1]

    def forward(self, pre_embedding, preembed_mask):
        if pre_embedding is None:
            return None, None
        split = True
        if preembed_mask is None:
            split = False
            preembed_mask = torch.zeros(pre_embedding.shape[:2], dtype=torch.bool, device=self.device)

        batch_of_seq = self.flat(pre_embedding)
        outputs = []
        Sp = 0
        for seq, mask in zip(batch_of_seq, preembed_mask):
            # seq is (S,input_dim)
            # mask is (S,)
            if not split:
                outputs.append(self.output_from_lstm(seq=seq, subseq=None).reshape((1, -1)))
                Sp = max(Sp, 1)
            else:
                applied_lstm = [
                    self.output_from_lstm(seq=seq, subseq=subseq)
                    for subseq in self.subseqs_from_mask(mask=mask)
                ]

                outputs.append(torch.stack(applied_lstm, dim=0))
                Sp = max(Sp, len(applied_lstm))
        ret = torch.zeros((len(batch_of_seq), Sp, self.embedding_dim),
                          device=self.device)
        mask = torch.ones((len(batch_of_seq), Sp),
                          dtype=torch.bool,
                          device=self.device,
                          )
        for i, output in enumerate(outputs):
            ret[i, :len(output)] = output
            mask[i, :len(output)] = False
        return ret, mask
