if __name__ == '__main__':
    import torch
    from matplotlib import pyplot as plt
    from BERTeam.trainer.team_trainer import DiscreteInputTrainer
    from BERTeam.buffer.language_replay_buffer import ReplayBufferDiskStorage

    # results:
    #  seems like just L1 loss does not work when the optimizer is complex
    #   (it gets stuck maximizing another guy, does not learn good conditional distributions)
    #  just BERT additional head:
    #   looks like it can learn to optimize, but it forgets sometimes and optimizes something else, leading to 'spikes'
    #  just entropy reg:
    #   it puts more distribution on optimal guy, but it seems to have difficulty learning languagy things
    #  both BERT head and entropy reg:
    #   seems to learn distribution in 'spikes' still, and the spikes grow to approach the correct distribution
    #   the learning does seem to be noisy, potentially a result of increased entropy or low embedding dim

    # torch.random.manual_seed(69)
    # L1 loss test for dual output layer
    E = 32  # embedding dim
    bert_loss = True  # whether to add BERT loss
    entropy_reg = 0.10  # how much to entropy regularize (0. does nothing)
    batch_size = None  # putting None here will always use all of buffer for a stable gradient, otherwise, we will sample
    epochs = 1500
    dropout = 0.1
    T = 3

    # team, weight
    data = [
        ((0, 0), 2.),
        ((0, 1), 2.),
        ((1, 0), 2.),
        ((1, 1), 1.),
    ]
    data = [
        ((0, 0, 0), 0.1),
        ((0, 0, 1), 0.2),
        ((0, 1, 0), 1.6),
        ((0, 1, 1), 0.3),
        ((1, 0, 0), 0.4),
        ((1, 0, 1), 0.5),
        ((1, 1, 0), 0.6),
        ((1, 1, 1), 0.7),
    ]
    keys = [t for t, w in data]

    test = DiscreteInputTrainer(
        num_agents=2,
        num_input_tokens=2,
        embedding_dim=E,
        pos_encode_teams=True,  # settting this to false makes it fail to distingish order
        append_pos_encode_teams=True,
        num_decoder_layers=2,
        num_encoder_layers=2,
        dropout=dropout,
        nhead=4,
        buffer=ReplayBufferDiskStorage(
            storage_dir='L1_test_replay_buffer',
            capacity=100,
            track_age=False,
        ),
        num_output_layers=2,
    )
    summed_data = dict()
    for t, w in data:
        if t not in summed_data:
            summed_data[t] = 0
        summed_data[t] += w
    opt = max([summed_data[t] for t in keys])
    best_responses = [t for t in keys if summed_data[t] == opt]
    for t, w in data:
        test.add_to_buffer(
            scalar=1,  # this is not usually the way to do it, however, this works for this example
            # usually, we would put scalar=w, and use a binned replay buffer so that BERTeam is trained to predict good teams
            # however, in our case we have very few points with all positive weights,
            #   so we can get away with just weighting them according to their output
            obs_mask=None,
            team=torch.tensor(t),
            obs_preembed=None,
            weight=w
        )
    losses = []
    prop_real = []
    for epoch in range(epochs):
        if bert_loss:
            # seems to help with learning to add BERT training on different head
            test.training_step_BERT(batch_size=batch_size,
                                    output_layer_idx=1,
                                    )
        l1loss = test.training_step_L1(batch_size=batch_size,
                                       entropy_reg=entropy_reg,
                                       )
        dist = test.get_total_distribution(T=T)
        dist2 = test.get_total_distribution(T=T, output_layer_idx=1)
        print('epoch', epoch)
        print('best: L1',
              tuple(round(dist[t], 2) for t in best_responses),
              'BERT',
              tuple(round(dist2[t], 2) for t in best_responses),
              )
        print('wors: L1',
              tuple(round(dist[t], 2) for t in keys if t not in best_responses),
              'BERT',
              tuple(round(dist2[t], 2) for t in keys if t not in best_responses),
              )
        print('loss: L1', round(l1loss, 2))
        losses.append(l1loss)
        prop_real.append(sum([dist[t] for t in best_responses]))
        print()
    test.clear()
    plt.plot(losses, label='L1 loss')
    plt.plot(prop_real, label='correct prop')
    plt.legend()
    plt.show()
