import numpy as np
import torch

from rewards import advantage, arm_reward


def test_float32_cast_of_float64_advantage_is_within_1e6():
    rng = np.random.default_rng(3)
    for _ in range(2000):
        g = rng.integers(0, 201, size=4).astype(float)
        for arm, w in (("graded", 5.0), ("binary", 5.0), ("binary", 5000.0), ("none", 0.0)):
            a = advantage(arm_reward(g, arm, w, 30.0))
            t = torch.tensor(a, dtype=torch.float32)
            assert float(np.max(np.abs(t.double().numpy() - a))) < 1e-6
