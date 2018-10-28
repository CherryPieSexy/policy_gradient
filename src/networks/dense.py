import torch.nn as nn


__all__ = [
    'DenseSeparate',
    'DenseShared'
]


class DenseActor(nn.Module):
    net_type = 'dense_actor'
    """Actor network with one head for policy estimation"""
    def __init__(self, obs_dim, hidden_dim, n_actions):
        super(DenseActor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(True),
            nn.Linear(hidden_dim, n_actions)
        )

    def forward(self, observation):
        return self.model(observation)


class DenseCritic(nn.Module):
    net_type = 'dense_critic'
    """Critic network with one head for value estimation"""
    def __init__(self, obs_dim, hidden_dim):
        super(DenseCritic, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(True),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, observation):
        return self.model(observation)


class DenseSeparate(nn.Module):
    net_type = 'dense_separate'
    """Actor-Critic network with tho heads with separate parameters"""
    def __init__(self, obs_dim, hidden_dim, n_actions):
        super(DenseSeparate, self).__init__()
        self.actor = DenseActor(obs_dim, hidden_dim, n_actions)
        self.critic = DenseCritic(obs_dim, hidden_dim)

    def forward(self, observation):
        log_policy = self.actor(observation)
        value = self.critic(observation)
        return log_policy, value


class DenseShared(nn.Module):
    net_type = 'dense_shared'
    """Actor-Critic network with two heads with shared parameters"""
    def __init__(self, obs_dim, hidden_dim, n_actions):
        super(DenseShared, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(True),
        )
        self.policy_layer = nn.Linear(hidden_dim, n_actions)
        self.value_layer = nn.Linear(hidden_dim, 1)

    def forward(self, observation):
        features = self.feature_extractor(observation)
        log_policy = self.policy_layer(features)
        value = self.value_layer(features)
        return log_policy, value
