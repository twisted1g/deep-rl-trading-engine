"""Загрузка SB3 модели + VecNormalize. Поддержка Dueling DQN."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal, Tuple

import numpy as np
from gymnasium import Env, spaces

from . import research_bridge  # noqa: F401

log = logging.getLogger(__name__)

Algo = Literal["a2c", "ppo", "dqn", "dueling_dqn"]


class _DummyObsEnv(Env):
    """Минимальный gym Env, нужен только чтобы VecNormalize.load мог обернуть
    его (он смотрит observation_space/action_space)."""

    metadata = {"render_modes": []}

    def __init__(self, obs_dim: int):
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)

    def reset(self, *, seed=None, options=None):
        return np.zeros(self.observation_space.shape, dtype=np.float32), {}

    def step(self, action):
        return (
            np.zeros(self.observation_space.shape, dtype=np.float32),
            0.0,
            True,
            False,
            {},
        )


def load_model_and_vecnorm(
    algo: Algo,
    model_path: str | Path,
    vecnorm_path: str | Path,
    obs_dim: int,
) -> Tuple[Any, Any]:
    """Возвращает (model, vec_env_with_normalize)."""
    from stable_baselines3 import A2C, DQN, PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    model_path = Path(model_path)
    vecnorm_path = Path(vecnorm_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not vecnorm_path.exists():
        raise FileNotFoundError(f"VecNormalize stats not found: {vecnorm_path}")

    vec_env = DummyVecEnv([lambda: _DummyObsEnv(obs_dim)])
    vec_env = VecNormalize.load(str(vecnorm_path), vec_env)
    vec_env.training = False
    vec_env.norm_reward = False

    custom_objects: dict = {}
    if algo == "dueling_dqn":
        from agents.dueling_dqn_policy import DuelingDQNPolicy

        custom_objects["policy_class"] = DuelingDQNPolicy

    algo_cls = {"a2c": A2C, "ppo": PPO, "dqn": DQN, "dueling_dqn": DQN}[algo]
    model = algo_cls.load(
        str(model_path), env=vec_env, custom_objects=custom_objects or None
    )
    log.info("Loaded %s from %s (vecnorm=%s)", algo, model_path, vecnorm_path)
    return model, vec_env


def predict_action(model: Any, vec_env: Any, raw_obs: np.ndarray) -> int:
    """raw_obs: 1D массив. Нормализует через VecNormalize и зовёт model.predict."""
    obs_batched = raw_obs.reshape(1, -1).astype(np.float32)
    obs_norm = vec_env.normalize_obs(obs_batched)
    action, _ = model.predict(obs_norm, deterministic=True)
    return int(np.asarray(action).flatten()[0])
