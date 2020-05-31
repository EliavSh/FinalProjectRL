import random
import torch


def init_q():
    pass


class ZombieMaster:
    def __init__(self, strategy, num_actions, device):
        self.current_action = 0  # TODO - delete because we don't need to save the action after we chose it, just pass it to the environment
        self.strategy = strategy
        self.device = device
        self.current_step = 0
        self.num_actions = num_actions

    def select_action(self, state, policy_net):
        rate = self.strategy.get_exploration_rate(self.current_step)
        self.current_step += 1

        # if rate > random.random():
        if True:
            action = random.randrange(self.num_actions)
            return torch.tensor([action]).to(self.device)  # explore
        else:
            with torch.no_grad():
                return policy_net(state).argmax(dim=1).to(self.device)  # exploit


"""    
def step(self):
        
        self.current_action = random.choice(self.env.start_positions)
        self.env.add_zombie(self.current_action)
        print("ZombieMaster action:", self.current_action)
"""
