__all__ = ['Reinforce']


class Reinforce:
    def __init__(self, environment, net, gamma=0.99):
        self.environment = environment
        self.net = net
        self.gamma = gamma

    def discounted_reward(self, rewards):
        d_reward = [rewards[-1]]
        for r in reversed(rewards[:-1]):
            d_reward.append(r + self.gamma * d_reward[-1])
        return d_reward[::-1]

    def train_step(self):
        observations, rewards, actions = self.play_episode()
        d_rewards = self.discounted_reward(rewards)
        # TODO: torch here
        return d_rewards
