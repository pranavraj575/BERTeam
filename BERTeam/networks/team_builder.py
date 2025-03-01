import torch
from torch import nn
from torch.nn import Transformer, Embedding

from BERTeam.networks.positional_encoder import IdentityEncoding, ClassicPositionalEncoding, PositionalAppender
from BERTeam.networks.input_embedding import InputEmbedder


class BERTeam(nn.Module):
    def __init__(self,
                 num_agents,
                 embedding_dim=512,
                 nhead=8,
                 num_encoder_layers=16,
                 num_decoder_layers=16,
                 dim_feedforward=None,
                 dropout=.1,
                 PosEncConstructor=PositionalAppender,
                 num_output_layers=1,
                 trans_kwargs=None,
                 pos_enc_kwargs=None,
                 ):
        """
        Args:
            num_agents: number of possible agents to choose from
            embedding_dim: transformer embedding dim
            nhead: numbwr of attention heads
            num_encoder_layers: number of encoder layers
            num_decoder_layers: number of decoder layers
            dim_feedforward: dimension feedforward
            dropout: dropout
            PosEncConstructor: what positional encoder to use
            num_output_layers:
            trans_kwargs: dict with any other transformer kwargs
            pos_enc_kwargs: dict with any positional encoding kwargs
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        if dim_feedforward is None:
            dim_feedforward = self.embedding_dim*4
        self.num_agents = num_agents
        self.num_tokens = num_agents + 3
        # adding [MASK], [CLS], and [CLS2] tokens ([CLS] is for target (team) sequences, [CLS2] is for inputs)
        self.CLS = num_agents
        self.CLS2 = num_agents + 1
        self.MASK = num_agents + 2
        self.agent_embedding = Embedding(num_embeddings=self.num_tokens,
                                         embedding_dim=embedding_dim,
                                         )
        peekwargs = {'d_model': embedding_dim,
                     'dropout': dropout,
                     }
        if pos_enc_kwargs is not None:
            peekwargs.update(pos_enc_kwargs)
        self.pos_encoder = PosEncConstructor(**peekwargs)
        tkwargs = {
            'd_model': embedding_dim,
            'nhead': nhead,
            'num_encoder_layers': num_encoder_layers,
            'num_decoder_layers': num_decoder_layers,
            'dim_feedforward': dim_feedforward,
            'batch_first': True,  # takes in (N,S,E) input where N is batch and S is seq length
            'dropout': dropout,
        }
        if trans_kwargs is not None:
            tkwargs.update(trans_kwargs)
        self.transform = Transformer(**tkwargs)
        self.output_layers = [nn.Linear(embedding_dim, num_agents) for _ in range(num_output_layers)]
        self.num_output_layers = num_output_layers
        self.softmax = nn.Softmax(dim=-1)
        self.primary_output_layer = 0

    def set_primary_output_layer(self, output_layer_idx):
        assert type(output_layer_idx) == int and output_layer_idx >= 0 and output_layer_idx < self.num_output_layers
        self.primary_output_layer = output_layer_idx

    def add_cls_tokens(self, target_team):
        """
        adds [CLS] tokens to target teams
        Args:
            target_team: (N, T) vector
        Returns:
            (N, T+1) vector where self.CLS is added to the beginning of every team
        """
        (N, T) = target_team.shape
        return torch.cat((torch.ones((N, 1), dtype=target_team.dtype)*self.CLS, target_team), dim=1)

    def forward(self,
                input_embedding,
                target_team,
                input_mask,
                output_probs=True,
                pre_softmax=False,
                output_layer_idx=None,
                ):
        """
        Args:
            input_embedding: (N, S, E) shape tensor of input, or None if no input
                S should be very small, probably the output of embeddings with a more efficient method like LSTM
            target_team: (N, T) shape tensor of team members
                EACH TEAM SHOULD BE WITH A [CLS] token
            input_mask: (N, S) tensor of booleans on whether to mask each input
            output_probs: whether to output the probability of each team member
                otherwise just outputs the final embedding
            pre_softmax: if True, does not apply softmax to logits
            output_layer_idx: if there are multiple output layers, specify which one to use
        Returns:
            (
            cls: (N,E) embedding for the whole team,
            output:
                if output_probs, (N, T, num_agents) probability distribution for each position
                otherwise, (N, T, embedding_dim) output of transformer model
            )
        """
        if output_layer_idx is None:
            output_layer_idx = self.primary_output_layer
        N, T = target_team.shape

        # creates a sequence of size S+1
        # add to input embedding (N,S,E) to get (N, S+1, E) where embeddings of [CLS2] tokens are the first values
        # we create an array sized (N,1,E), where all N of the tokens are [CLS2]
        cls2_tokens = self.agent_embedding(torch.fill(torch.zeros((N, 1), dtype=target_team.dtype), self.CLS2))
        if input_embedding is None:
            # in this case, S=0, so our desried (N,1,E) array is just N copies of the [CLS2] tokens
            S = 0
            source = cls2_tokens
        else:
            # in this case, we concatenate the (N,S,E) array with the (N,1,E) array to get (N,S+1,E)
            N, S, _ = input_embedding.shape
            source = torch.cat((input_embedding, cls2_tokens,), dim=1)
        # update input mask from (N,S) boolean array to (N,S+1) to match the extra [CLS2] token
        if input_mask is None:
            # if no input mask is given, start by masking nothing (all zeros)
            input_mask = torch.zeros((N, S), dtype=torch.bool)
        # concatenate (N,S) array with (N,1) to get (N,S+1), we also append all zeros since we dont want to mask [CLS2]
        input_mask = torch.cat((input_mask, torch.zeros((N, 1), dtype=torch.bool)), dim=1)

        target = self.agent_embedding(self.add_cls_tokens(target_team))
        pos_enc_target = self.pos_encoder(target)
        # shaped (N,T+1,E), with [CLS] token at the beginning

        model_output = self.transform.forward(src=source,
                                              tgt=pos_enc_target,
                                              src_key_padding_mask=input_mask,
                                              memory_key_padding_mask=input_mask,
                                              )
        # (N,T+1,E), same as tgt

        # split into cls (N,E) and output corresponding to sequence (N,T,E)
        cls, output = model_output[:, 0, :], model_output[:, 1:, :]
        if output_probs:
            # if we want a distribution, apply the correct output layer onto the output embedding
            output_layer = self.output_layers[output_layer_idx]
            output = output_layer.forward(output)
            if not pre_softmax:
                output = self.softmax.forward(output)

        return cls, output


class TeamBuilder(nn.Module):
    def __init__(self,
                 input_embedder: InputEmbedder,
                 berteam: BERTeam,
                 ):
        """
        one of berteam or num_agents must be defined
        Args:
            input_embedder:
            berteam:
            num_agents:
        """
        super().__init__()
        self.input_embedder = input_embedder
        self.berteam = berteam
        self.num_output_layers = self.berteam.num_output_layers

    def forward(self,
                obs_preembed,
                target_team,
                obs_mask,
                output_probs=True,
                pre_softmax=False,
                output_layer_idx=None,
                ):
        """
        Args:
            obs_preembed: (N, S, *) shape tensor of input, or None if no input
            target_team: (N, T) shape tensor of team members
            obs_mask: (N, S) tensor of booleans on whether to mask each input
            output_probs: whether to output the probability of each team member
                otherwise just outputs the final embedding
            pre_softmax: if True, does not apply softmax to logits
            output_layer_idx: if specified, use this output layer
                else use the default (0)
        Returns: same as BERTeam.forward
            (
            cls: (N,E) embedding for the whole team,
            output:
                if output_probs, (N, T, num_agents) probability distribution for each position
                otherwise, (N, T, embedding_dim) output of transformer model
            )
        """
        if obs_preembed is None:
            obs_embed = None
            embed_mask = None
        else:
            obs_embed, embed_mask = self.input_embedder(obs_preembed, obs_mask)
        return self.berteam.forward(input_embedding=obs_embed,
                                    target_team=target_team,
                                    input_mask=embed_mask,
                                    output_probs=output_probs,
                                    pre_softmax=pre_softmax,
                                    output_layer_idx=output_layer_idx,
                                    )


if __name__ == '__main__':
    N = 10
    S = 7
    T = 5
    E = 16

    test = BERTeam(num_agents=69, embedding_dim=E, dropout=0)  # remove randomness

    team = torch.randint(0, test.num_agents, (N, T))

    obs_preembeda = torch.rand((N, S, E))
    # mask everything except for first element
    obs_mask = torch.ones((N, S), dtype=torch.bool)
    obs_mask[:, 0] = False

    # use/dont use obs_mask
    clsa, outputa = test.forward(input_embedding=obs_preembeda,
                                 target_team=team,
                                 input_mask=obs_mask,
                                 )
    clsa_unmsk, outputa_unmsk = test.forward(input_embedding=obs_preembeda,
                                             target_team=team,
                                             input_mask=None,
                                             )

    # everything is different except for first element
    obs_preembedb = torch.rand((N, S, E))
    obs_preembedb[:, 0] = obs_preembeda[:, 0]
    clsb, outputb = test.forward(input_embedding=obs_preembedb,
                                 target_team=team,
                                 input_mask=obs_mask,
                                 )
    clsb_unmsk, outputb_unmsk = test.forward(input_embedding=obs_preembedb,
                                             target_team=team,
                                             input_mask=None,
                                             )
    # the masked values should be the same
    assert torch.all(outputa == outputb)
    assert torch.all(clsa == clsb)

    # the unmasked values do not have to be the same
    print('expected false:', torch.all(outputa_unmsk == outputb_unmsk).item())
    print('expected false:', torch.all(clsa_unmsk == clsb_unmsk).item())
