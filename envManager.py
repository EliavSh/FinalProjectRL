import numpy as np
import math
import random
import gameGrid
import zombie
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T


"""
Assumptions - data types:
angle: float in range [-zombie_max_angle, zombie_max_angle] radians
current position of zombies: int in range [0, grid_height * grid_width - 1]

the zombie home is positioned in the middle of the left side of the board.
"""


class EnvManager:
    def __init__(self, env, device):
        self.device = device
        self.env = env
        self.env.reset()
        self.current_screen = None
        self.done = False

    def reset(self):
        self.env.reset()
        self.current_screen = None

    def num_actions_available(self):
        return self.env.action_space()

    def take_action(self, action_zombie, action_light):
        _, reward, self.done = self.env.step(action_zombie, action_light)  # TODO - here is used to be action.item() instead of just action, need to see how its affect us
        return torch.tensor([reward], device=self.device)

    def just_starting(self):
        # current screen is set to none in the beggining and in the end of an episode
        return self.current_screen is None

    def get_state(self):
        if self.just_starting() or self.done:
            self.current_screen = self.get_processed_screen()
            black_screen = torch.zeros_like(self.current_screen)
            return black_screen
        else:
            s1 = self.current_screen
            s2 = self.get_processed_screen()
            self.current_screen = s2
            return s2 - s1

    def get_screen_height(self):
        screen = self.get_processed_screen()
        return screen.shape[2]

    def get_screen_width(self):
        screen = self.get_processed_screen()
        return screen.shape[3]

    def get_processed_screen(self):
        screen = self.env.get_state().transpose((2, 0, 1))  # PyTorch expects CHW
        screen = self.crop_screen(screen)
        return self.transform_screen_data(screen)

    def crop_screen(self, screen):
        screen_height = screen.shape[1]

        # Strip off top and bottom
        top = int(screen_height * 0)
        bottom = int(screen_height * 1)
        screen = screen[:, top:bottom, :]
        return screen

    def transform_screen_data(self, screen):
        # Convert to float, rescale, convert to tensor
        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
        screen = torch.from_numpy(screen)

        # Use torchvision package to compose image transforms
        resize = T.Compose([
            T.ToPILImage()
            , T.Resize((80, 40))
            , T.ToTensor()
        ])

        return resize(screen).unsqueeze(0).to(self.device)  # add a batch dimension (BCHW)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
