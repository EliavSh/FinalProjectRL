import random
import torch


def init_q():
    pass


class LightMaster:
    def __init__(self, strategy, num_actions, device):
        self.strategy = strategy
        self.device = device
        self.current_step = 0
        self.num_actions = num_actions
        self.smart_actions = []
        self.random_actions = []

    def select_action(self, state, policy_net):
        rate = self.strategy.get_exploration_rate(self.current_step)
        self.current_step += 1

        if rate > random.random():
            action = random.randrange(self.num_actions)
            self.random_actions.append(action)
            return torch.tensor([action]).to(self.device)  # explore
        else:
            with torch.no_grad():
                policy_net_state = policy_net(state)
                policy_net_state_argmax = policy_net_state.argmax(dim=0)
                self.smart_actions.append(policy_net(state).argmax(dim=0).data.cpu().numpy()[0])
                return torch.tensor([policy_net(state).argmax(dim=0).data.cpu().numpy()[0]]).to(self.device)  # exploit

"""
plot smart actions:
    import matplotlib.pyplot as plt
    plt.hist(self.smart_actions)
    plt.show()
"""