"""
Gymnasium wrapper: random product category on each episode reset.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import gymnasium as gym
import numpy as np


class RandomCategoryResetWrapper(gym.Wrapper):
    """
    On each reset, sample a product category unless ``options['product_category']`` is set.

    The wrapped PricingEnv must accept ``product_category`` in reset options.
    """

    def __init__(
        self,
        env: gym.Env,
        categories: Sequence[str],
        seed: Optional[int] = None,
    ):
        super().__init__(env)
        self.categories: List[str] = list(categories)
        self._rng = np.random.default_rng(seed)
        self.last_category: Optional[str] = None

    def reset(self, *, seed=None, options=None):
        options = dict(options or {})
        if "product_category" not in options:
            options["product_category"] = str(
                self._rng.choice(self.categories)
            )
        self.last_category = options["product_category"]
        return self.env.reset(seed=seed, options=options)
