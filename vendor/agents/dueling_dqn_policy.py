from __future__ import annotations

from typing import List

import torch as th
from torch import nn

from stable_baselines3.dqn.policies import DQNPolicy, QNetwork


class DuelingQNetwork(QNetwork):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        in_dim = self.features_dim
        layers: List[nn.Module] = []
        for hidden in self.net_arch:
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(self.activation_fn())
            in_dim = hidden

        action_dim = int(self.action_space.n)
        self.trunk = nn.Sequential(*layers)
        self.value_head = nn.Linear(in_dim, 1)
        self.advantage_head = nn.Linear(in_dim, action_dim)
        self.q_net = nn.Identity()

    def forward(self, obs: th.Tensor) -> th.Tensor:
        features = self.extract_features(obs, self.features_extractor)
        x = self.trunk(features)
        v = self.value_head(x)
        a = self.advantage_head(x)
        return v + a - a.mean(dim=1, keepdim=True)


class DuelingDQNPolicy(DQNPolicy):
    def make_q_net(self) -> DuelingQNetwork:
        net_args = self._update_features_extractor(self.net_args, features_extractor=None)
        return DuelingQNetwork(**net_args).to(self.device)
