import unittest
import numpy as np
import gameGrid
import zombie
import envManager


class TestZombie(unittest.TestCase):

    def setUp(self):
        self.env = envManager.EnvManager(gameGrid.Grid(10, 5), np.pi / 10, 1, 1, 2)
        self.zombie = zombie.Zombie(id=0, angle=0, velocity=1, y=5, env=self.env)

    def test_step_x(self):
        x_before = self.zombie.x
        self.zombie.step()
        self.assertEqual(self.zombie.x, x_before + 1)

    def test_step_y(self):
        y_before = self.zombie.y
        self.zombie.step()
        self.assertEqual(self.zombie.y, y_before)
