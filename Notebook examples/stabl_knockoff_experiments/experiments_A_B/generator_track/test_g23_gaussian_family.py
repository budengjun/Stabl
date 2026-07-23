from __future__ import annotations

import unittest

import numpy as np

from research_common import GeneratorConfig, estimate_factor_covariance


class G23GaussianFamilyTests(unittest.TestCase):
    def test_factor_config_requires_rank(self) -> None:
        with self.assertRaises(ValueError):
            GeneratorConfig.from_mapping(
                {
                    "label": "bad",
                    "generator": "gaussian_factor_mvr",
                }
            )

    def test_factor_covariance_is_symmetric_positive_definite(self) -> None:
        rng = np.random.default_rng(8)
        X = rng.normal(size=(50, 12))
        Sigma = estimate_factor_covariance(
            X,
            rank=3,
            diag_floor=1e-3,
            shrinkage=0.05,
        )
        self.assertEqual(Sigma.shape, (12, 12))
        self.assertTrue(np.allclose(Sigma, Sigma.T))
        self.assertGreater(float(np.linalg.eigvalsh(Sigma).min()), 0.0)

    def test_sdp_config_is_supported(self) -> None:
        config = GeneratorConfig.from_mapping(
            {"label": "sdp", "generator": "gaussian_sdp"}
        )
        self.assertEqual(config.generator, "gaussian_sdp")


if __name__ == "__main__":
    unittest.main()
