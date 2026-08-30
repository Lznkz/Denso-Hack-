import unittest

import numpy as np
import torch

from src.phase1.core.data import load_cmapss_files, normalize_engines
from src.phase1.core.model import SmartConv1DAutoencoder
from src.phase1.core.windowing import cut_windows_for_engine
from src.phase1.main import DATA_PATHS, split_train_val


class PipelineSmokeTest(unittest.TestCase):
    def test_data_to_model_forward(self):
        engines = load_cmapss_files(DATA_PATHS)
        train, val = split_train_val(engines, val_fraction=0.2, seed=0)
        train, val = normalize_engines(train[:2], val[:1])
        windows = cut_windows_for_engine(train[0], window_len=20, stride=10)

        self.assertEqual(len({engine.unit_number for engine in engines}), len(engines))
        self.assertFalse({engine.unit_number for engine in train} & {engine.unit_number for engine in val})
        self.assertEqual(train[0].X.shape[1], 9)
        self.assertEqual(train[0].W.shape[1], 3)
        self.assertTrue(np.isfinite(train[0].X).all())

        model = SmartConv1DAutoencoder(n_sensors=9, window_len=20, w_dim=3, z_dim=4)
        x_hat, w_hat, latent = model(torch.from_numpy(windows.X[:2]))
        self.assertEqual(tuple(x_hat.shape), tuple(windows.X[:2].shape))
        self.assertEqual(tuple(w_hat.shape), (2, 3))
        self.assertEqual(tuple(latent.shape), (2, 4))


if __name__ == "__main__":
    unittest.main()
