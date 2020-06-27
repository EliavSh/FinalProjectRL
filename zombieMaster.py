import random
import torch


def init_q():
    pass


class ZombieMaster:
    def __init__(self, strategy, num_actions, device):
        self.strategy = strategy
        self.device = device
        self.current_step = 0
        self.num_actions = num_actions

    def select_action(self, state, policy_net):
        rate = self.strategy.get_exploration_rate(self.current_step)
        self.current_step += 1
        random_number = random.random()
        if rate > random_number:  # TODO - replace it with the 'if True' condition, only after we succeed with the basic learning of the light master
            # if True:
            action = random.randrange(self.num_actions)
            return torch.tensor([action]).to(self.device)  # explore
        else:
            with torch.no_grad():
                # here we are getting the action from one pass along the network. after that we:
                # convert the tensor to data, then move to cpu using then converting to numpy and lastly, wraping back to tensor
                action = torch.tensor([policy_net(state).argmax(dim=1).data.cpu().numpy()[0]]).to(self.device)
                if action.ndim == 0:
                    error = 404
                return action
