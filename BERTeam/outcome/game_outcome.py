import torch


class PlayerInfo:
    """
    structure for player observations of a multi-agent game
    """

    def __init__(self, obs_preembed=None, obs_mask=None):
        """
        Args:
            obs_preembed: (S,*) array of a sequence of observations
                None if no observations
            obs_mask: (S,) boolean array of which observations to mask
                None if no mask
        """
        self.obs_preembed = obs_preembed
        self.obs_mask = obs_mask
        self.S = 0 if obs_preembed is None else obs_preembed.shape[0]

    def clone(self):
        return PlayerInfo(obs_preembed=None if self.obs_preembed is None else self.obs_preembed.clone(),
                          obs_mask=None if self.obs_mask is None else self.obs_mask.clone(),
                          )

    def get_data(self, reshape=True):
        """
        gets data
        Args:
            reshape: whtether to return in batch format
        Returns:
            self.obs_preembed,self.obs_mask
        """
        if reshape:
            return (None if self.obs_preembed is None else self.obs_preembed.unsqueeze(0),
                    None if self.obs_mask is None else self.obs_mask.unsqueeze(0))
        else:
            return self.obs_preembed, self.obs_mask

    def union_obs(self,
                  other_player_info,
                  combine=True,
                  ):
        """
        creates clone, combining observations with other_obs and other_mask

        Args:
            other_player_info: other player info with obs and mask either None or shape (S',*) and (S',)
                if self.obs_preembed has shape (S,*), the two * dimensions must be the same
            combine: whether to combine or just return updates
        Returns:
            concatenates the two preembeddings and masks
            either outputs an outcome with None, None or
                (S+1+S', *) concatenated output, (S+1+S',) boolean mask
                 an extra masked element is added in the middle to denote the division
                or other_obs_preembed and other_mask (if not chain_observations)
        """
        other_obs_preembed = other_player_info.obs_preembed
        other_mask = other_player_info.obs_mask
        if not combine:
            return PlayerInfo(obs_preembed=other_obs_preembed,
                              obs_mask=other_mask,
                              )
        Spp = self.S + 1 + other_player_info.S

        if Spp == 1:
            # in this case, there are no observations and no masks, so just return empty observation
            return PlayerInfo()

        if (self.obs_preembed is not None) and (other_obs_preembed is not None):
            # if both are not None, we append the observations,  size (S+1+S', *)

            # divider is shape (1,*)
            divider = torch.zeros_like(self.obs_preembed[:1])
            new_preembed = torch.cat((self.obs_preembed, divider, other_obs_preembed), dim=0)

            # we must also set the mask
            # by default mask nothing except the divider
            new_mask = torch.zeros(Spp, dtype=torch.bool)
            new_mask[self.S] = True  # always mask the division

            if self.obs_mask is not None:
                new_mask[:self.S] = self.obs_mask
            if other_mask is not None:
                new_mask[self.S + 1:] = other_mask
        else:
            # in this case, one is empty, the other has observations
            # just return the one that is nonempty
            if self.obs_preembed is not None:
                new_preembed = self.obs_preembed.clone()
                new_mask = self.obs_mask
            else:
                new_preembed = other_obs_preembed
                new_mask = other_mask
        return PlayerInfo(
            obs_preembed=new_preembed,
            obs_mask=new_mask,
        )

    def __str__(self):
        return ('PlayerInfo(' +
                'obs_preembed:' + str(self.obs_preembed) + '; ' +
                'obs_mask:' + str(self.obs_mask) + '; ' +
                ')')

    def __eq__(self, other):
        if self.obs_preembed is None or other.obs_preembed is None:
            return (self.obs_preembed is None) and (other.obs_preembed is None)

        mask_equality = (
                (self.obs_mask is None and other.obs_mask is None) or
                (self.obs_mask is None and torch.sum(other.obs_mask) == 0) or
                (other.obs_mask is None and torch.sum(self.obs_mask) == 0) or
                torch.equal(self.obs_mask, other.obs_mask)
        )
        return torch.equal(self.obs_preembed, other.obs_preembed) and mask_equality


class OutcomeFn:
    """
    structure for calculating outcomes of a team game
    gives an outcome score for each team, all outcome scores are non-negative and sum to 1
        usually 1 is win, 0.5 is tie (for two team games) and 0 is loss
    """

    def __init__(self):
        super().__init__()
        self.ident = 0
        self.dir = None

    def get_outcome(self, team_choices, agent_choices, **kwargs):
        """
        Args:
            team_choices: K-tuple of teams, each team is an array of players
            agent_choices: same shape as team_choices, calculated agents (if applicable)

        Returns: list corresponding to teams
            [
                outcome score,
                list corresponding to players of PlayerInfo(
                    obs_preembed=player observation (None or size (S,*) seq of observations);
                    obs_mask=observation mask (None or size (S,) boolean array of which items to mask;
                    )
                list can be empty, this will correspond to an empty observation
            ]
        """
        raise NotImplementedError

    def set_ident(self, ident):
        self.ident = ident

    def set_dir(self, dir):
        self.dir = dir


if __name__ == '__main__':
    test0 = PlayerInfo()
    test = PlayerInfo(obs_preembed=torch.rand((1, 2, 3)))
    test2 = PlayerInfo(obs_preembed=torch.rand((2, 2, 3)))
    print(test0)
    print(test)
    print(test0.union_obs(test))
    print(test.union_obs(test))
    print(test.union_obs(test2))
    print(test0 == test)
    print(test0.union_obs(test) == test)
    print(test.union_obs(test0) == test)
