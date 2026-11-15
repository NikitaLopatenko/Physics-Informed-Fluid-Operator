import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.fno import FNO2d


class FNOTests(unittest.TestCase):
    def test_same_weights_run_on_a_finer_grid(self):
        torch.manual_seed(0)
        model = FNO2d(modes=8, width=8, layers=2)
        coarse = model(torch.randn(2, 1, 32, 32))
        fine = model(torch.randn(2, 1, 64, 64))
        self.assertEqual(tuple(coarse.shape), (2, 1, 32, 32))
        self.assertEqual(tuple(fine.shape), (2, 1, 64, 64))
        self.assertLess(float(coarse.detach().mean().abs()), 1e-5)
        self.assertLess(float(fine.detach().mean().abs()), 1e-5)

    def test_one_step_reaches_the_fourier_weights(self):
        torch.manual_seed(0)
        model = FNO2d(modes=4, width=8, layers=2)
        pred = model(torch.randn(2, 1, 16, 16))
        loss = (pred - torch.randn(2, 1, 16, 16)).pow(2).mean()
        loss.backward()
        weight = model.spectral[0].weight_pos
        self.assertIsNotNone(weight.grad)
        self.assertGreater(float(weight.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
