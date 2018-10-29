import torch.nn as nn


__all__ = [
    'ConvSeparate',
    'ConvShared'
]

flat_size = 64 * 7 * 7


class ConvActor(nn.Module):
    net_type = 'conv_actor'
    """Actor network with one head for policy estimation"""
    def __init__(self, num_actions):
        super(ConvActor, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4), nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(True)
        )
        self.mlp = nn.Sequential(
            nn.Linear(flat_size, 512), nn.ReLU(True),
            nn.Linear(512, num_actions)
        )

    def forward(self, observation):
        conv_features = self.conv(observation)
        log_policy = self.mlp(conv_features.view(-1, flat_size))
        return log_policy


class ConvCritic(nn.Module):
    net_type = 'conv_critic'
    """Critic network with one head for value estimation"""
    def __init__(self):
        super(ConvCritic, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4), nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(True)
        )
        self.mlp = nn.Sequential(
            nn.Linear(flat_size, 512), nn.ReLU(True),
            nn.Linear(512, 1)
        )

    def forward(self, observation):
        conv_features = self.conv(observation)
        value = self.mlp(conv_features.view(-1, flat_size))
        return value


class ConvSeparate(nn.Module):
    net_type = 'conv_separate'
    """Actor-Critic network with two heads with separate parameters"""
    def __init__(self, num_actions):
        super(ConvSeparate, self).__init__()
        self.actor = ConvActor(num_actions)
        self.critic = ConvCritic()

    def forward(self, observation):
        log_policy = self.actor(observation)
        value = self.critic(observation)
        return log_policy, value


class ConvShared(nn.Module):
    net_type = 'conv_shared'
    """Actor-Critic network with two heads with shared parameters"""
    def __init__(self, num_actions):
        super(ConvShared, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4), nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(True)
        )
        self.mlp = nn.Sequential(
            nn.Linear(flat_size, 512), nn.ReLU(True),
        )
        self.policy_layer = nn.Linear(512, num_actions)
        self.value_layer = nn.Linear(512, 1)

    def forward(self, observation):
        conv_features = self.conv(observation)
        flat_features = self.mlp(conv_features.view(-1, flat_size))
        log_policy = self.policy_layer(flat_features)
        value = self.value_layer(flat_features)
        return log_policy, value
