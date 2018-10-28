import torch.nn as nn


__all__ = [
    'ActorCriticSeparate',
    'ActorCriticShared'
]


class ActorNet(nn.Module):
    net_type = 'dense_actor'
    """Actor network with one head for policy estimation"""
    def __init__(self, obs_dim, hidden_dim, n_actions):
        super(ActorNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(True),
            nn.Linear(hidden_dim, n_actions)
        )

    def forward(self, observation):
        return self.model(observation)


class CriticNet(nn.Module):
    net_type = 'dense_critic'
    """Critic network with one head for value estimation"""
    def __init__(self, obs_dim, hidden_dim):
        super(CriticNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(True),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, observation):
        return self.model(observation)


class ActorCriticSeparate(nn.Module):
    net_type = 'separate+actor_critic'
    """Actor-Critic network with tho heads with separate parameters"""
    def __init__(self, obs_dim, hidden_dim, n_actions):
        super(ActorCriticSeparate, self).__init__()
        self.actor = ActorNet(obs_dim, hidden_dim, n_actions)
        self.critic = CriticNet(obs_dim, hidden_dim)

    def forward(self, observation):
        log_policy = self.actor(observation)
        value = self.critic(observation)
        return log_policy, value


class ActorCriticShared(nn.Module):
    net_type = 'shared_actor_critic'
    """Actor-Critic network with two heads with shared parameters"""
    def __init__(self, obs_dim, hidden_dim, n_actions):
        super(ActorCriticShared, self).__init__()
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
