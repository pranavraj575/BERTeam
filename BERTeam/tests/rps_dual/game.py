import torch
from matplotlib import pyplot as plt
from BERTeam.outcome import PlayerInfo, OutcomeFn

ROCK = 0
PAPER = 1
SCISOR = 2


def double_game_outcome(a, b):
    # ['RR', 'PP', 'SS', 'RP', 'RS', 'PS']
    a, b = int(a), int(b)

    if a < 3 and b < 3:
        if a == b:
            return 0
        elif (a - b)%3 == 1:
            # i wins
            return 1
        else:
            return -1
    if a >= 3 and b >= 3:
        # ['RR', 'PP', 'SS', 'RP', 'RS', 'PS']
        # mapped to
        # ['PS', 'RS', 'RP', ...]
        # equivalent to
        # [ R P S ]
        return double_game_outcome(5 - a, 5 - b)
    elif ({a, b} in ({0, 4}, {1, 3}, {2, 5})):
        if a < b:
            return 1
        else:
            return -1
    else:
        if a > b:
            return 1
        else:
            return -1


class DualPreMappedOutcome(OutcomeFn):
    def __init__(self, agents):
        super().__init__()
        self.agents = agents

    def get_outcome(self, team_choices, **kwargs):
        result = double_game_outcome(team_choices[0], team_choices[1])
        if result == 0:
            return [(.5, [PlayerInfo(obs_preembed=team_choices[1]),
                          PlayerInfo(obs_preembed=team_choices[1])]),
                    (.5, [PlayerInfo(obs_preembed=team_choices[0]),
                          PlayerInfo(obs_preembed=team_choices[0])]),
                    ]
        if result == 1:
            return [(1, [PlayerInfo(obs_preembed=team_choices[1]),
                         PlayerInfo(obs_preembed=team_choices[1])]),
                    (0, [PlayerInfo(obs_preembed=team_choices[0]),
                         PlayerInfo(obs_preembed=team_choices[0])]),
                    ]
        if result == -1:
            return [(0, [PlayerInfo(obs_preembed=team_choices[1]),
                         PlayerInfo(obs_preembed=team_choices[1])]),
                    (1, [PlayerInfo(obs_preembed=team_choices[0]),
                         PlayerInfo(obs_preembed=team_choices[0])]),
                    ]


class DualPairOutcome(OutcomeFn):

    def map_to_number(self, team):
        # ['RR', 'PP', 'SS', 'RP', 'RS', 'PS']
        team = team.flatten()
        if team[0] == team[1]:
            return team[0]
        team = sorted(team)
        if team == [0, 1]:
            return torch.tensor(3)
        if team == [1, 2]:
            return torch.tensor(5)
        return torch.tensor(4)

    def get_outcome(self, team_choices, **kwargs):
        result = double_game_outcome(self.map_to_number(team_choices[0]),
                                     self.map_to_number(team_choices[1]),
                                     )
        if result == 0:
            return [(.5, [PlayerInfo(obs_preembed=team_choices[1]),
                          PlayerInfo(obs_preembed=team_choices[1])]),
                    (.5, [PlayerInfo(obs_preembed=team_choices[0]),
                          PlayerInfo(obs_preembed=team_choices[0])]),
                    ]
        if result == 1:
            return [(1, [PlayerInfo(obs_preembed=team_choices[1]),
                         PlayerInfo(obs_preembed=team_choices[1])]),
                    (0, [PlayerInfo(obs_preembed=team_choices[0]),
                         PlayerInfo(obs_preembed=team_choices[0])]),
                    ]
        if result == -1:
            return [(0, [PlayerInfo(obs_preembed=team_choices[1]),
                         PlayerInfo(obs_preembed=team_choices[1])]),
                    (1, [PlayerInfo(obs_preembed=team_choices[0]),
                         PlayerInfo(obs_preembed=team_choices[0])]),
                    ]


if __name__ == '__main__':
    torch.random.manual_seed(69)
    agents = torch.arange(6)
    outcomes = DualPreMappedOutcome(agents=agents)
    print(outcomes.get_outcome(team_choices=(torch.tensor([3]), torch.tensor([0]))))
    print(outcomes.get_outcome(team_choices=(torch.tensor([3]), torch.tensor([1]))))
    print(outcomes.get_outcome(team_choices=(torch.tensor([3]), torch.tensor([2]))))
    print()
    outcomes = DualPairOutcome(agents=agents)
    print(outcomes.get_outcome(team_choices=(torch.tensor([2, 1]), torch.tensor([1, 1]))))
