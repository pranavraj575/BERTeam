import torch
from BERTeam.trainer import DiscreteInputTrainer
from BERTeam.buffer import ReplayBufferDiskStorage

import numpy as np
import itertools

ROCK = 0
PAPER = 1
SCISOR = 2


def dist_from_trainer(trainer: DiscreteInputTrainer,
                      input_preembedding=None,
                      input_mask=None,
                      num_agents=3,
                      keyorder=None
                      ):
    trainer.team_builder.eval()
    dic = dict()
    keys = []
    for choice in itertools.chain(itertools.combinations(range(num_agents), 1),
                                  itertools.combinations(range(num_agents), 2)):
        if len(choice) == 1:
            choice = choice + choice
            perms = [choice]
        else:
            perms = itertools.permutations(choice)
        keys.append(choice)
        prob = 0
        for perm in perms:
            orders = list(itertools.permutations(range(2)))
            for order in orders:
                this_prob = 1.
                target_team = trainer.create_masked_teams(T=2, N=1)

                for idx in order:
                    full_dist = trainer.team_builder.forward(input_preembedding, target_team, input_mask)
                    dist = full_dist[0, idx]
                    this_prob *= dist[perm[idx]].item()
                    target_team[0, idx] = perm[idx]
                prob += this_prob/len(orders)
        dic[choice] = prob

    trainer.team_builder.train()
    if keyorder is None:
        keyorder = keys
    return np.array([dic[key] for key in keyorder])


if __name__ == '__main__':
    import os, sys
    from BERTeam.tests.rps_basic.teams import plot_dist_evolution, loss_plot
    from BERTeam.tests.rps_dual.game import DualPairOutcome
    import time

    torch.random.manual_seed(69)
    agents = torch.arange(3)

    DIR = os.getcwd()
    plot_dir = os.path.join(DIR, 'data', 'plots', 'tests_rps2_teams')
    if not os.path.exists((plot_dir)):
        os.makedirs(plot_dir)
    trainer = DiscreteInputTrainer(num_agents=3,
                                   num_input_tokens=3,
                                   embedding_dim=64,
                                   pos_encode_input=True,
                                   append_pos_encode_input=True,
                                   pos_encode_teams=True,
                                   append_pos_encode_teams=True,
                                   num_decoder_layers=4,
                                   num_encoder_layers=4,
                                   buffer=ReplayBufferDiskStorage(
                                       storage_dir=os.path.join(DIR, "data", "temp", "tests_rps2_teams"),
                                       capacity=int(2e4),
                                       device=None,
                                   ),
                                   )

    N = 100
    minibatch = 64
    init_dists = []
    cond_dists = []
    strat_labels = ['RR', 'PP', 'SS', 'RP', 'RS', 'PS']
    losses = []
    outcome = DualPairOutcome()
    for epoch in range(100):
        start_time = time.time()
        noise = trainer.create_nose_model_towards_uniform(1/np.sqrt(epoch/2 + 1))
        game, _, _ = trainer.create_teams(T=2, N=2, noise_model=noise)

        trainer.buffer.add_outcome(teams=game,
                                   outcome=outcome.get_outcome(team_choices=game),
                                   filter=lambda score: score > .5,
                                   )

        init_distribution = dist_from_trainer(trainer=trainer,
                                              input_preembedding=None,
                                              input_mask=None,
                                              )
        init_dists.append(init_distribution)
        conditional_dists = []
        for opponent_choice in itertools.chain(itertools.combinations(range(3), 1),
                                               itertools.combinations(range(3), 2)):
            if len(opponent_choice) == 1:
                opponent_choice = opponent_choice + opponent_choice
            dist = dist_from_trainer(trainer=trainer,
                                     input_preembedding=torch.tensor([opponent_choice]),
                                     input_mask=None,
                                     )
            conditional_dists.append(dist)
        cond_dists.append(conditional_dists)

        collection_time = time.time() - start_time
        start_time = time.time()
        loss = trainer.train(batch_size=minibatch)
        losses.append(loss)

        print('epoch', epoch, ';\tbuffer size', len(trainer.buffer))
        print('\tcollection time:', round(collection_time, 2))
        print('\ttrain time:', round(time.time() - start_time, 2))

        print('\tloss', loss)

        if not (epoch + 1)%10:
            start_time = time.time()
            loss_plot(losses, save_dir=os.path.join(plot_dir, 'loss_plot.png'))

            plot_dist_evolution(init_dists,
                                save_dir=os.path.join(plot_dir, 'init_dist.png'),
                                labels=strat_labels
                                )
            for k, name in enumerate(strat_labels):
                plot_dist_evolution([dist[k] for dist in cond_dists],
                                    save_dir=os.path.join(plot_dir, 'dist_against' + name + '.png'),
                                    title='Distribution against ' + name,
                                    labels=strat_labels,
                                    )
            print('\tplot time:', round(time.time() - start_time, 2))
        epoch += 1
        print()
    trainer.clear()
