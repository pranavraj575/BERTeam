import torch, os
from torch import nn
from torch.utils.data import DataLoader

from BERTeam.buffer import LangReplayBuffer

from BERTeam.networks import TeamBuilder, BERTeam
from BERTeam.networks import DiscreteInputEmbedder, DiscreteInputPosEmbedder, DiscreteInputPosAppender
from BERTeam.networks import IdentityEncoding, ClassicPositionalEncoding, PositionalAppender


class TeamTrainer:
    def __init__(self, num_agents, MASK=-1):
        self.num_agents = num_agents
        self.MASK = MASK

    def reset_storage_dir(self, storage_dir):
        pass

    def clear(self):
        """
        clears any memory/data
        """
        pass

    def save(self, save_dir):
        pass

    def load(self, save_dir):
        pass

    def add_to_buffer(self, scalar, obs_preembed, team, obs_mask, weight=1.):
        """
        adds element to language replay buffer
        Args:
            scalar:
            obs_preembed:
            team:
            obs_mask:

        Returns:

        """
        pass

    def uniform_random_team(self, shape):
        return torch.randint(0, self.num_agents, shape)

    def get_total_distribution(self,
                               T,
                               N=1,
                               init_team=None,
                               tracked=None,
                               **kwargs,
                               ):
        raise NotImplementedError

    def get_member_distribution(self,
                                init_team,
                                indices=None,
                                noise_model=None,
                                **kwargs
                                ):
        """
        returns probability distribution of next member being added to team
        Args:
            init_team: already initialized team
            indices: indices to check, if None, checks all masked elements
        Returns:
            indices, distribution of agents for each index
                for this class, uniform
        """

        if indices is None:
            indices = torch.where(torch.eq(init_team, self.MASK))

        og_dist = torch.ones((len(indices[0]), self.num_agents))/self.num_agents
        if noise_model is not None:
            # add noise if this is a thing
            dist = noise_model(og_dist)
        else:
            dist = og_dist.clone()
        return indices, dist, og_dist

    def add_member_to_team(self,
                           member,
                           T=None,
                           N=1,
                           init_team=None,
                           team_noise_model=None,
                           valid_locations=None,
                           **kwargs,
                           ):
        """
        adds the specified member to the team in a random position (sampled from the network distribution)
        one of T or init_team must be specified
        Args:
            member: member to add (index)
            T: team size
            N: number of teams to make (default 1)
            init_team: team to add member team
                if None, starts with all masks
            team_noise_model: noise model that takes T-element multinomial distributions and returns another (with noise)
                ((N,T) -> (N,T))
                default None
            valid_locations: boolean array of size (N,T) that determines whether each location is valid
        Returns:
            team with member attached
        """
        if init_team is None:
            init_team = self.create_masked_teams(T=T, N=N)

        # (N, T, num_agents)
        dist = torch.ones((N, T, self.num_agents))/self.num_agents

        # (N, T)
        conditional_dist = dist[:, :, member]

        if team_noise_model is not None:
            conditional_dist = team_noise_model(conditional_dist)

        # set all non-masked or invalid entries to 0
        conditional_dist[torch.where(torch.not_equal(init_team, self.MASK))] = 0
        if valid_locations is not None:
            conditional_dist = conditional_dist*valid_locations  # sets invalid locations to 0
            conditional_dist = conditional_dist/torch.sum(conditional_dist, dim=-1, keepdim=True)

        valid_indices = torch.where(torch.sum(conditional_dist, dim=1) > 0)[0]
        if len(valid_indices) == 0:
            # there are no mask tokens, or the conditional distribution is all zeros
            return init_team

        # samples from conditional_dist at the valid indices
        places = torch.multinomial(conditional_dist[valid_indices,], 1).flatten()

        init_team[valid_indices, places] = member
        return init_team

    def fill_in_teams(self,
                      initial_teams,
                      noise_model=None,
                      num_masks=None,
                      valid_members=None,
                      **kwargs
                      ):
        """
        replaces all [MASK] with team members
            repeatedly calls mutate_add_member
        Args:
            initial_teams: initial torch array of team members, shape (N,T)
            obs_preembed: input to give input embedder, size (N,S,*)
                None if no input
            obs_mask: size (N,S) boolean array of whether to mask each input embedding
            noise_model: noise to add to probability distribution, if None, doesnt add noise
            num_masks: specify the largest number of [MASK] tokens in any row
                if None, just calls mutate_add_member (T) times
            valid_members: (N,T,num_agents) boolean array of which agents are valid for which locations
        Returns:
            filled in teams of size (N,T), log formation probability, log noiseless foramtion probability
        """
        N, T = initial_teams.shape
        if num_masks is None:
            num_masks = T
        log_formation_prob = torch.tensor(0.)
        og_log_formation_prob = torch.tensor(0.)
        for _ in range(num_masks):
            initial_teams, lfp, lofp = self.mutate_add_member(
                initial_teams=initial_teams,
                indices=None,
                noise_model=noise_model,
                valid_members=valid_members,
                **kwargs,
            )
            log_formation_prob = log_formation_prob + lfp
            og_log_formation_prob = og_log_formation_prob + lofp
        return initial_teams, log_formation_prob, og_log_formation_prob

    def mutate_add_member(self,
                          initial_teams,
                          indices=None,
                          noise_model=None,
                          valid_members=None,
                          **kwargs,
                          ):
        """
        updates initial teams by updating specified indices with samples from the probability distribution
            if indices is None, chooses indices that correspond with [MASK] tokens (one for each row)
        Args:
            initial_teams: initial torch array of team members, shape (N,T)
            indices: indices to replace (if None, picks one masked index at random)
                should be in pytorch format, (list of dim 0 indices, list of dim 1 indices)
            obs_preembed: input to give input embedder, size (N,S,*)
                None if no input
            obs_mask: size (N,S) boolean array of whether to mask each input embedding
            noise_model: noise to add to probability distribution, if None, doesnt add noise
            valid_members: (N,T,num_agents) boolean array of which agents are valid for which locations
        Returns:
            team with updates, formation probability, noiseless foramtion probability
        """
        N, T = initial_teams.shape
        if indices is None:
            indices = [[], []]
            for i in range(N):
                potential = torch.where(torch.eq(initial_teams[i], self.MASK))[0]
                if len(potential) > 0:
                    indices[0].append(i)
                    indices[1].append(potential[torch.randint(0, len(potential), (1,))])
                # else:
                #    print("warning: tried to add to full team")
        if len(indices[0]) == 0:
            # no [MASK] tokens exist
            return initial_teams, torch.tensor(0.), torch.tensor(0.)
        og_dist = torch.ones((N, T, self.num_agents))/self.num_agents

        if noise_model is not None:
            # add noise if this is a thing
            dist = noise_model(og_dist)
        else:
            dist = og_dist.clone()

        if valid_members is not None:
            dist = dist*valid_members
            dist = dist/torch.sum(dist, dim=-1, keepdim=True)

            og_dist = og_dist*valid_members
            og_dist = og_dist/torch.sum(og_dist, dim=-1, keepdim=True)
        # just look at the relevant indices
        dist = dist[indices]

        # torch.multinomial samples each
        result = torch.multinomial(dist,
                                   len(indices[0]),
                                   replacement=True,
                                   ).flatten()
        initial_teams[indices] = result
        log_formation_probability = torch.sum(torch.log(dist[indices[0], indices[1], result])).detach()
        log_og_formation_probability = torch.sum(torch.log(og_dist[indices[0], indices[1], result])).detach()
        return initial_teams, log_formation_probability, log_og_formation_probability

    def create_masked_teams(self, T, N=1):
        """
        Args:
            T: team size
            N: batch size
        Returns:
            an (N,T) tensor of all masked members
        """
        return torch.fill(torch.zeros((N, T), dtype=torch.long), self.MASK)

    def train(self, *args, **kwargs):
        pass

    def create_nose_model_towards_uniform(self, t):
        """
        creates a noise model that takes in a distribution and does a weighted average with a uniform dist
        Args:
            t: weight to give uniform dist (1.0 replaces the distribution, 0.0 leaves dist unchanged)
                or array of length (K,), doing it on individual values
        Returns:
            distribution -> distribution map, which are sized (N,K) for an N batch of a K-multinomial
        """

        def noise(dist):
            dist = (1 - t)*dist + (t)*torch.ones_like(dist)/dist.shape[-1]
            return dist/torch.sum(dist, dim=-1, keepdim=True)

        return noise

    def create_teams(self,
                     T,
                     N=1,
                     obs_preembed=None,
                     obs_mask=None,
                     noise_model=None,
                     ):

        """
        creates random teams of size (N,T)
        Args:
            T: number of team members
            N: number of teams to make (default 1)
            obs_preembed: input to give input embedder, size (N,S,*)
                None if no input
            obs_mask: size (N,S) boolean array of whether to mask each input embedding
            noise_model: noise to add to probability distribution, if None, doesnt add noise
        Returns:
            filled in teams of size (N,T), formation probability, noiseless foramtion probability
        """
        return self.uniform_random_team(shape=(N, T)), torch.tensor(1/N*T), torch.tensor(1/N*T)


class MLMTeamTrainer(TeamTrainer):
    def __init__(self,
                 team_builder: TeamBuilder,
                 buffer: LangReplayBuffer,
                 storage_dir=None,
                 weight_decay_half_life=None,
                 optimizer_kwargs=None,
                 ):
        """
        Args:
            team_builder:
            buffer:
            storage_dir: place to store files
                if None, sets later
            weight_decay_half_life: we would like to exponentially decay weights as time goes on
                if we track age (self.buffer.track_age), we can do this
                 weight_decay_half_life is the half life of the weights
                 None if no decay
        """
        super().__init__(
            num_agents=team_builder.berteam.num_agents,
            MASK=team_builder.berteam.MASK,
        )
        self.team_builder = team_builder
        self.buffer = buffer
        self.weight_decay_half_life = weight_decay_half_life

        if storage_dir is not None:
            self.reset_storage_dir(storage_dir=storage_dir)

        if optimizer_kwargs is None:
            optimizer_kwargs = dict()
        self.optim = torch.optim.Adam(team_builder.parameters(), **optimizer_kwargs)

    def reset_storage_dir(self, storage_dir):
        self.buffer.reset_storage_dir(storage_dir=os.path.join(storage_dir, 'buffer'))

    def clear(self):
        """
        clears any memory/data
        """
        super().clear()
        self.buffer.clear()

    def save(self, save_dir):
        super().save(save_dir=save_dir)
        self.buffer.save(os.path.join(save_dir, 'buffer'))
        dic = {'model': self.team_builder.state_dict(),
               'optim': self.optim.state_dict(),
               }
        torch.save(dic, os.path.join(save_dir, 'model.pkl'))

    def load(self, save_dir):
        super().load(save_dir=save_dir)
        self.buffer.load(os.path.join(save_dir, 'buffer'))
        dic = torch.load(os.path.join(save_dir, 'model.pkl'))
        self.team_builder.load_state_dict(dic['model'])
        self.optim.load_state_dict(dic['optim'])

    def add_to_buffer(self, scalar, obs_preembed, team, obs_mask, weight=1.):
        item = (scalar, obs_preembed, team, obs_mask, weight)
        self.buffer.push(item=item)

    def train(self,
              batch_size,
              minibatch=None,
              mask_probs=None,
              replacement_probs=(.8, .1, .1),
              mask_obs_prob=.1,
              ):
        if minibatch is None:
            minibatch = batch_size
        if not len(self.buffer):
            print('WARNING: trying to train on empty buffer')
            return None
        loss = 0
        for i in range(0, batch_size, minibatch):
            tinybatch = min(i + minibatch, batch_size) - i
            # cut off the last batch if we go over
            step_loss = self.training_step(
                batch_size=tinybatch,
                mask_probs=mask_probs,
                replacement_probs=replacement_probs,
                mask_obs_prob=mask_obs_prob,
            )
            loss += tinybatch*step_loss
        # clear any remaining gradient from memory
        self.optim.zero_grad()

        # torch.clear_autocast_cache()
        # if torch.cuda.is_available():
        #    torch.cuda.empty_cache()
        return loss/batch_size

    def training_step(self,
                      batch_size,
                      mask_probs=None,
                      replacement_probs=(.8, .1, .1),
                      mask_obs_prob=.1,
                      ):
        """
        Args:
            batch_size: size of batch to train on
            mask_probs: list of proabilities of masking to sample from
                if None, samples (1/team_size, 2/team_size, ..., 1)
                    for each obtained team size
            replacement_probs: proportion of ([MASK], random element, same element) to replace masked elements with
                default (.8, .1, .1) because BERT
            mask_obs_prob: if >0, randomly masks observations with this probability as well
        Returns:
            avg losses for whole dataset
        """
        if not len(self.buffer):
            print('WARNING: trying to train on empty buffer')
            return None
        data = self.buffer.sample(batch=batch_size)
        self.optim.zero_grad()
        losses = torch.zeros(1)
        count = 0
        for item in data:
            if self.buffer.track_age:
                (scalar, obs_preembed, team, obs_mask, weight), age = item
                if self.weight_decay_half_life is not None:
                    weight = weight*torch.pow(torch.tensor(1/2), age/self.weight_decay_half_life)
            else:
                (scalar, obs_preembed, team, obs_mask, weight) = item
            team = team.view((1, -1))
            if torch.is_tensor(obs_preembed) and torch.all(torch.isnan(obs_preembed)):
                obs_preembed = None
            if torch.is_tensor(obs_mask) and torch.all(torch.isnan(obs_mask)):
                obs_mask = None
            if mask_probs is None:
                N, T = team.shape
                mask_probs = torch.arange(0, T + 1)/T

            mask_prob = mask_probs[torch.randint(0, len(mask_probs), (1,))]

            loss = self._mask_and_learn(obs_preembed=obs_preembed,
                                        teams=team,
                                        obs_mask=obs_mask,
                                        mask_prob=mask_prob,
                                        replacement_probs=replacement_probs,
                                        mask_obs_prob=mask_obs_prob,
                                        )
            # keep gradients
            losses += loss*weight
            count += 1
        mean_loss = losses/count
        mean_loss.backward()
        self.optim.step()
        return mean_loss.item()

    def _mask_and_learn(self,
                        obs_preembed,
                        teams,
                        obs_mask,
                        mask_prob,
                        replacement_probs=(.8, .1, .1),
                        mask_obs_prob=.1,
                        ):
        """
        runs learn step on a lot of mask_probabilities
            need to test with a wide range of mask probabilites because for generation, we may start with a lot of masks
        Args:
            obs_preembed: tensor (N, S, *) of input preembeddings, or None if no preembeddings
            teams: tensor (N, T) of teams
            obs_mask: boolean tensor (N, S) of whether to pad each input
            mask_prob: probabilty of masking to use
            replacement_probs: proportion of ([MASK], random element, same element) to replace masked elements with
                default (.8, .1, .1) because BERT
            mask_obs_prob: if >0, randomly mask observations with this prob
        Returns:
            avg crossentropy loss
        """
        if mask_obs_prob > 0 and obs_preembed is not None:
            if obs_mask is not None:
                temp_obs_mask = obs_mask.clone()
            else:
                temp_obs_mask = torch.zeros((obs_preembed.shape[:2]))
            additional_mask = torch.rand_like(temp_obs_mask, dtype=torch.float) < mask_obs_prob
            temp_obs_mask = torch.logical_or(temp_obs_mask, additional_mask)
        else:
            temp_obs_mask = obs_mask
        loss = self._get_losses(obs_preembed=obs_preembed,
                                teams=teams,
                                obs_mask=temp_obs_mask,
                                mask_prob=mask_prob,
                                replacement_probs=replacement_probs,
                                )
        return loss

    def _get_losses(self,
                    obs_preembed,
                    teams,
                    obs_mask,
                    mask_prob=.5,
                    replacement_probs=(.8, .1, .1),
                    ):
        """
        randomly masks winning team members, runs the transformer token prediction model, and gets crossentropy loss
            of predicting the masked tokens
        Args:
            obs_preembed: tensor (N, S, *) of input preembeddings, or None if no preembeddings
            teams: tensor (N, T) of teams
            obs_mask: boolean tensor (N, S) of whether to pad each input
            mask_prob: proportion of elements to mask (note that we will always mask at least one per batch
            replacement_probs: proportion of ([MASK], random element, same element) to replace masked elements with
                default (.8, .1, .1) because BERT
        Returns:
            crossentropy loss of prediction
        """

        in_teams, mask, mask_indices = self._randomly_mask_teams(teams=teams,
                                                                 mask_prob=mask_prob,
                                                                 replacement_props=replacement_probs,
                                                                 )
        logits = self.team_builder.forward(obs_preembed=obs_preembed,
                                           target_team=in_teams,
                                           obs_mask=obs_mask,
                                           output_probs=True,
                                           pre_softmax=True,
                                           )
        criterion = nn.CrossEntropyLoss()

        loss = criterion(logits[mask_indices], teams[mask_indices])
        return loss

    def _randomly_mask_teams(self, teams, mask_prob, replacement_props):
        """
        Args:
            teams: size (N,T) tensor to randomly mask
            mask_prob: proportion of elements to mask (note that we will always mask at least one per batch)
            replacement_props: proportion of ([MASK], random element, same element) to replace masked elements with
                tuple of three floats
        """
        masked_teams = teams.clone()
        N, T = teams.shape

        which_to_mask = torch.bernoulli(torch.ones_like(teams)*mask_prob)
        forced_mask = torch.randint(0, T, (N,))
        which_to_mask[torch.arange(N), forced_mask] = 1

        mask_indices = torch.where(torch.eq(which_to_mask, 1))
        num_masks = len(mask_indices[0])
        what_to_replace_with = torch.multinomial(torch.tensor([replacement_props for _ in range(num_masks)]),
                                                 1).flatten()
        # this will be a tensor of 0s, 1s, and 2s, indicating whether to replace each mask with
        #   [MASK], a random element, or not to replace it

        # map this to -1,-2,-3 respectively
        what_to_replace_with = -1 - what_to_replace_with
        # now map to the actual values to replace with in the array
        what_to_replace_with = torch.masked_fill(what_to_replace_with,
                                                 mask=torch.eq(what_to_replace_with, -1),
                                                 value=self.MASK,
                                                 )
        rand_rep = torch.where(torch.eq(what_to_replace_with, -2))
        what_to_replace_with[rand_rep] = self.uniform_random_team((len(rand_rep[0]),))

        unchange = torch.where(torch.eq(what_to_replace_with, -3))
        # grab the correct team members from the original array
        what_to_replace_with[unchange] = masked_teams[[dim_idx[unchange] for dim_idx in mask_indices]]

        masked_teams[mask_indices] = what_to_replace_with
        return masked_teams, which_to_mask, mask_indices

    def add_member_to_team(self,
                           member,
                           T=None,
                           N=1,
                           init_team=None,
                           obs_preembed=None,
                           obs_mask=None,
                           team_noise_model=None,
                           valid_locations=None,
                           ):
        """
        adds the specified member to the team in a random position (sampled from the network distribution)
        one of T or init_team must be specified
        Args:
            member: member to add (index)
            T: team size
            N: number of teams to make (default 1)
            init_team: team to add member team
                if None, starts with all masks
            obs_preembed: observation to condition on
            obs_mask: mask for observations
            team_noise_model: noise model that takes T-element multinomial distributions and returns another (with noise)
                ((N,T) -> (N,T))
                default None
            valid_locations: boolean array of size (N,T) that determines whether each location is valid
        Returns:
            team with member attached
        """
        if init_team is None:
            init_team = self.create_masked_teams(T=T, N=N)

        # (N, T, num_agents)
        dist = self.team_builder.forward(obs_preembed=obs_preembed,
                                         target_team=init_team,
                                         obs_mask=obs_mask,
                                         output_probs=True,
                                         pre_softmax=False,
                                         )
        # (N, T)
        conditional_dist = dist[:, :, member]

        if team_noise_model is not None:
            conditional_dist = team_noise_model(conditional_dist)

        # set all non-masked entries to 0
        conditional_dist[torch.where(init_team != self.MASK)] = 0
        if valid_locations is not None:
            conditional_dist = conditional_dist*valid_locations
            conditional_dist = conditional_dist/torch.sum(conditional_dist, dim=-1, keepdim=True)

        valid_indices = torch.where(torch.sum(conditional_dist, dim=1) > 0)[0]
        if len(valid_indices) == 0:
            # there are no mask tokens, or the conditional distribution is all zeros
            return init_team

        # samples from conditional_dist at the valid indices
        places = torch.multinomial(conditional_dist[valid_indices,], 1).flatten()

        init_team[valid_indices, places] = member
        return init_team

    def create_teams(self,
                     T,
                     N=1,
                     obs_preembed=None,
                     obs_mask=None,
                     noise_model=None,
                     ):

        """
        creates random teams of size (N,T)
        Args:
            T: number of team members
            N: number of teams to make (default 1)
            obs_preembed: input to give input embedder, size (N,S,*)
                None if no input
            obs_mask: size (N,S) boolean array of whether to mask each input embedding
            noise_model: noise to add to probability distribution, if None, doesnt add noise
        Returns:
            filled in teams of size (N,T), formation probability, noiseless foramtion probability
        """
        return self.fill_in_teams(initial_teams=self.create_masked_teams(T=T, N=N),
                                  obs_preembed=obs_preembed,
                                  obs_mask=obs_mask,
                                  noise_model=noise_model,
                                  )

    def get_total_distribution(self,
                               T,
                               N=1,
                               init_team=None,
                               prob=torch.tensor(1.),
                               tracked=None,
                               obs_preembed=None,
                               obs_mask=None,
                               noise_model=None,
                               valid_members=None,
                               ):
        if init_team is None:
            init_team = self.create_masked_teams(T=T, N=N)
        if tracked is None:
            tracked = dict()

        indices, dist, og_dist = self.get_member_distribution(init_team=init_team,
                                                              indices=None,
                                                              obs_preembed=obs_preembed,
                                                              obs_mask=obs_mask,
                                                              noise_model=noise_model,
                                                              valid_members=valid_members,
                                                              )
        if indices is None:
            obj = tuple(init_team.detach().numpy().flatten())
            if obj not in tracked:
                tracked[obj] = 0
            tracked[obj] += prob.item()
        else:
            for i, idx in enumerate(zip(*indices)):
                for member, value in enumerate(dist[i]):
                    temp_team = init_team.clone()
                    temp_team[idx] = member
                    # probability of sleceting this is prob of selecting init_team
                    # times prob of selecting this index (1/ number of indices)
                    # times transformer prob

                    selection_prob = value*prob/len(indices[0])
                    self.get_total_distribution(T=T,
                                                N=N,
                                                init_team=temp_team,
                                                prob=selection_prob,
                                                tracked=tracked,
                                                obs_preembed=obs_preembed,
                                                obs_mask=obs_mask,
                                                noise_model=noise_model,
                                                valid_members=valid_members,
                                                )
        return tracked

    def get_member_distribution(self,
                                init_team,
                                indices=None,
                                obs_preembed=None,
                                obs_mask=None,
                                noise_model=None,
                                valid_members=None,
                                ):
        if indices is None:
            indices = torch.where(torch.eq(init_team, self.MASK))
        if len(indices[0]) == 0:
            return None, torch.tensor(1.), torch.tensor(1.)
        output = self.team_builder.forward(obs_preembed=obs_preembed,
                                           target_team=init_team,
                                           obs_mask=obs_mask,
                                           output_probs=True,
                                           pre_softmax=False,
                                           )

        og_dist = output[indices]  # (|indices|, num_agents) multinomial distribution for each index to update
        if noise_model is not None:
            # add noise if this is a thing
            dist = noise_model(og_dist)
        else:
            dist = og_dist.clone()
        if valid_members is not None:
            # set invalid members to 0
            dist = dist*(valid_members[indices])
            dist = dist/torch.sum(dist, dim=-1, keepdim=True)

            og_dist = og_dist*(valid_members[indices])
            og_dist = og_dist/torch.sum(og_dist, dim=-1, keepdim=True)
        return indices, dist.detach(), og_dist.detach()

    def mutate_add_member(self,
                          initial_teams,
                          indices=None,
                          obs_preembed=None,
                          obs_mask=None,
                          noise_model=None,
                          valid_members=None,
                          ):
        """
        updates initial teams by updating specified indices with samples from the probability distribution
            if indices is None, chooses indices that correspond with [MASK] tokens (one for each row)
        Args:
            initial_teams: initial torch array of team members, shape (N,T)
            indices: indices to replace (if None, picks one masked index at random)
                should be in pytorch format, (list of dim 0 indices, list of dim 1 indices)
            obs_preembed: input to give input embedder, size (N,S,*)
                None if no input
            obs_mask: size (N,S) boolean array of whether to mask each input embedding
            noise_model: noise to add to probability distribution, if None, doesnt add noise
            valid_members: (N,T,num_agents) boolean array of which agents are valid for which locations
        Returns:
            team with updates, formation probability, noiseless foramtion probability
        """
        N, T = initial_teams.shape
        if indices is None:
            indices = [[], []]
            for i in range(N):
                potential = torch.where(initial_teams[i] == self.MASK)[0]
                if len(potential) > 0:
                    indices[0].append(i)
                    indices[1].append(potential[torch.randint(0, len(potential), (1,))])
                # else:
                #    print("warning: tried to add to full team")
        if len(indices[0]) == 0:
            # no [MASK] tokens exist
            return initial_teams, torch.tensor(0.), torch.tensor(0.)
        indices, dist, og_dist = self.get_member_distribution(init_team=initial_teams,
                                                              indices=indices,
                                                              obs_preembed=obs_preembed,
                                                              obs_mask=obs_mask,
                                                              noise_model=noise_model,
                                                              valid_members=valid_members
                                                              )

        # torch.multinomial samples each
        result = torch.multinomial(dist,
                                   len(indices[0]),
                                   replacement=True,
                                   ).flatten()
        initial_teams[indices] = result
        log_formation_probability = torch.sum(torch.log(dist[range(len(dist)), result])).detach()
        log_og_formation_probability = torch.sum(torch.log(og_dist[range(len(og_dist)), result])).detach()
        return initial_teams, log_formation_probability, log_og_formation_probability


class DiscreteInputTrainer(MLMTeamTrainer):
    def __init__(self,
                 buffer: LangReplayBuffer,
                 num_agents,
                 num_input_tokens,
                 embedding_dim=512,
                 nhead=8,
                 num_encoder_layers=16,
                 num_decoder_layers=16,
                 dim_feedforward=None,
                 dropout=.1,
                 pos_encode_input=True,
                 append_pos_encode_input=False,
                 pos_encode_teams=True,
                 append_pos_encode_teams=False,
                 weight_decay_half_life=100,
                 optimizer_kwargs=None,
                 ):
        if pos_encode_teams:
            if append_pos_encode_teams:
                PosEncConstructorTeams = PositionalAppender
            else:
                PosEncConstructorTeams = ClassicPositionalEncoding
        else:
            PosEncConstructorTeams = IdentityEncoding
        berteam = BERTeam(num_agents=num_agents,
                          embedding_dim=embedding_dim,
                          nhead=nhead,
                          num_encoder_layers=num_encoder_layers,
                          num_decoder_layers=num_decoder_layers,
                          dim_feedforward=dim_feedforward,
                          dropout=dropout,
                          PosEncConstructor=PosEncConstructorTeams,
                          )
        if pos_encode_input:
            if append_pos_encode_input:
                Constructor = DiscreteInputPosAppender
            else:
                Constructor = DiscreteInputPosEmbedder
            input_embedder = Constructor(num_embeddings=num_input_tokens,
                                         embedding_dim=embedding_dim,
                                         dropout=dropout,
                                         )
        else:
            input_embedder = DiscreteInputEmbedder(num_embeddings=num_input_tokens,
                                                   embedding_dim=embedding_dim,
                                                   )

        super().__init__(
            buffer=buffer,
            team_builder=TeamBuilder(
                berteam=berteam,
                input_embedder=input_embedder,
            ),
            weight_decay_half_life=weight_decay_half_life,
            optimizer_kwargs=optimizer_kwargs,
        )


if __name__ == '__main__':
    from matplotlib import pyplot as plt

    N = 32  # number of teams to train on per epoch
    eval_N = 9  # number of teams to evaluate on
    S = 2  # sequence length of input
    T = 5  # size of team
    num_inputs = 4  # number of distinct input tokens

    epochs = 300
    torch.random.manual_seed(69)

    E = 64  # embedding dim

    test = DiscreteInputTrainer(num_agents=69,
                                num_input_tokens=num_inputs,
                                embedding_dim=E,
                                pos_encode_input=True,
                                append_pos_encode_input=True,
                                pos_encode_teams=True,
                                append_pos_encode_teams=True,
                                num_decoder_layers=3,
                                num_encoder_layers=3,
                                dropout=0.1,
                                nhead=8,
                                )

    basic_team = torch.arange(T, dtype=torch.long)


    def random_shuffle():
        return basic_team[torch.randperm(len(basic_team))]


    def correct_output(input_pre):
        # test it on shifted inputs, mod some number
        # return (basic_team + input_pre[0])%4
        return (basic_team + torch.sum(input_pre))%4
        # return basic_team


    losses = []
    for epoch in range(epochs):
        # input_preembedding_gen = (torch.randint(0, num_inputs, (i%S,)) if i > 0 else None for i in range(N))
        # input_embedding_gen = (None for i in range(N))

        input_mask = [torch.nan for _ in range(N)]
        input_preembedding = torch.randint(0, num_inputs, (N, S))
        # input_mask = torch.zeros((N, S), dtype=torch.bool)
        # out_teams = torch.stack([random_shuffle() for _ in range(N)], dim=0)
        out_teams = torch.stack([correct_output(t) for t in input_preembedding], dim=0)

        data = list(zip(input_preembedding, out_teams, input_mask))
        loader = DataLoader(data,
                            shuffle=True,
                            batch_size=64,
                            )
        loss = test.training_step(
            data=loader,
            sgd=True
        )

        losses.append(loss)
        print('epoch', epoch, '\tloss', loss.item())
    # print(test.mask_and_learn(input_embedding_gen=input_embeddings,
    #                          winning_team_gen=(init_team.clone() for _ in range(N))))
    init_teams = test.create_masked_teams(T=T, N=eval_N)
    print(init_teams)
    # input_pre = torch.arange(0, num_teams).view((-1, 1))%num_inputs

    input_preembedding = torch.randint(0, num_inputs, (eval_N, S))
    for i in range(T):
        # force it to add member i to the team
        # init_teams = test.add_member_to_team(i, init_team=init_teams, obs_preembed=input_preembedding)

        test.mutate_add_member(initial_teams=init_teams, obs_preembed=input_preembedding)
        print(init_teams)
    print('goal:')
    goal = torch.stack([correct_output(t) for t in input_preembedding])
    print(goal)

    print('accuracy:',
          round((torch.sum(init_teams == goal)/torch.sum(torch.ones_like(init_teams))).item()*100, 2),
          '%')
    plt.plot(range(len(losses)), losses)
    plt.show()
