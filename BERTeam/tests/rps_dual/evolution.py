import torch
from src.coevolver import CaptianCoevolution
from coevolution_tests.rps_basic.game import plot_dist_evolution
from coevolution_tests.rps_dual.game import DualPreMappedOutcome

from src.team_trainer import TeamTrainer

if __name__ == '__main__':
    import os, sys

    torch.random.manual_seed(69)

    DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.join(os.getcwd(), sys.argv[0]))))
    plot_dir = os.path.join(DIR, 'data', 'plots', 'tests_rps2_evolution')
    if not os.path.exists((plot_dir)):
        os.makedirs(plot_dir)

    popsize = 100
    agents = torch.randint(0, 6, (popsize,))


    def clone(init, replacement):
        global agents
        temp = agents.clone()
        agents[init] = temp[replacement]


    def mutate(p=.05):
        global agents
        mask = torch.rand(popsize) < p
        agents[mask] = torch.randint(0, 6, (torch.sum(mask).item(),))


    trainer = CaptianCoevolution(population_sizes=[popsize],
                                 outcome_fn_gen=lambda: DualPreMappedOutcome(agents=agents),
                                 clone_fn=clone,
                                 mutation_fn=mutate,
                                 team_trainer=TeamTrainer(num_agents=popsize),
                                 captian_elo_update=.5,
                                 )

    init_dists = []
    for epoch in range(500):
        for i in range(10):
            trainer.epoch(rechoose=False)
        trainer.breed()
        trainer.mutate()
        init_distribution = [len(torch.where(agents == i)[0])/popsize for i in range(6)]
        init_dists.append(init_distribution)

        if not (epoch + 1)%10:
            plot_dist_evolution(init_dists, save_dir=os.path.join(plot_dir, 'init_dist.png'),
                                labels=['RR', 'PP', 'SS', 'RP', 'RS', 'PS'])
            print(epoch + 1)
