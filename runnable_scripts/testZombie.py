import time
import dask

from dask.distributed import Client

client = Client(asynchronous=True)


# import unittest
# class TestZombie(unittest.TestCase):
#     pass

def costly_simulation(list_params: list) -> int:
    total_score = 0
    for pp in list_params:
        time.sleep(pp)
        total_score += pp
    return total_score


total_scores = []
start = time.time()
# second try using client compute:
for param in [1, 2, 3, 2, 1, 3, 5, 1, 3, 5, 3, 5]:
    lazy_score = client.submit(costly_simulation, [param])
    total_scores.append(await lazy_score)

"""
first try using dask.compute
for param in [1, 2, 3, 2, 1, 3, 5, 1, 3, 5, 3, 5]:
    lazy_score = dask.delayed(costly_simulation)([param])
    total_scores.append(lazy_score)

total_scores = dask.compute(*total_scores)
"""
stop = time.time()

print(total_scores)
print(stop - start)

"""
    def setUp(self):
        self.env = envManager.EnvManager(gameGrid.Grid(10, 5), np.pi / 10, 1, 1, 2)
        self.zombie = zombie.Zombie(id=0, angle=0, velocity=1, y=5, env=self.env)

    def test_step_x(self):
        x_before = self.zombie.x
        self.zombie.move()
        self.assertEqual(self.zombie.x, x_before + 1)

    def test_step_y(self):
        y_before = self.zombie.y
        self.zombie.move()
        self.assertEqual(self.zombie.y, y_before)
"""
