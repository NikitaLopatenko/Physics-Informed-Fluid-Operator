import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.fno3d import FNO3d


class FNO3dTests(unittest.TestCase):
    def test_one_pass_runs_on_two_meshes(self):
        torch.manual_seed(0)
        model = FNO3d(modes=2, width=4, layers=2, out_channels=8)
        coarse = model(torch.randn(1, 3, 8, 8, 8))
        fine = model(torch.randn(1, 3, 16, 16, 16))
        self.assertEqual(tuple(coarse.shape), (1, 8, 8, 8, 8))
        self.assertEqual(tuple(fine.shape), (1, 8, 16, 16, 16))

    def test_fourier_weights_receive_a_gradient(self):
        torch.manual_seed(0)
        model = FNO3d(modes=2, width=4, layers=2, out_channels=4)
        loss = (model(torch.randn(1, 3, 8, 8, 8)) ** 2).mean()
        loss.backward()
        grad = model.spectral[0].weights[0].grad
        self.assertIsNotNone(grad)
        self.assertGreater(float(grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
