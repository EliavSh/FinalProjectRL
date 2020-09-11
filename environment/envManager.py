import numpy as np
import torch
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
        _, reward, self.done = self.env.step(action_zombie, action_light)
        return torch.tensor([reward], device=self.device)

    def just_starting(self):
        # current screen is set to none in the beginning and in the end of an episode
        return self.current_screen is None

    def get_state(self):
        zombie_grid = self.env.grid.get_values()
        zombie_grid = zombie_grid.astype(np.float32)
        zombie_grid.fill(0)
        health_grid = np.copy(zombie_grid)
        for i in self.env.alive_zombies:
            zombie_grid[int(i.y), int(i.x)] = 1
            health_grid[int(i.y), int(i.x)] = i.hit_points
        return torch.from_numpy(zombie_grid).flatten(), torch.from_numpy(np.concatenate((zombie_grid, health_grid))).flatten()

    def get_state_old(self):
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
        return self.env.grid.get_height()

    def get_screen_width(self):
        return self.env.grid.get_width()

    def get_processed_screen(self):
        screen = self.env.get_pygame_window().transpose((2, 0, 1))  # PyTorch expects CHW
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
            , T.Resize((60, 30))
            , T.ToTensor()
        ])

        return resize(screen).unsqueeze(0).to(self.device)  # add a batch dimension (BCHW)

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
