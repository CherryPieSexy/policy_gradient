from tqdm import trange
import torch

__all__ = ['Reinforce']


class Reinforce:
    agent = 'Reinforce'
    """Reinforce agent"""
    def __init__(self, environment,
                 net, optimizer, entropy_reg, writer,
                 gamma=0.99):
        self.environment = environment
        self.net = net
        self.optimizer = optimizer
        self.entropy_reg = entropy_reg
        self.writer = writer
        self.gamma = gamma

    def discounted_reward(self, rewards):
        d_reward = [rewards[-1]]
        for r in reversed(rewards[:-1]):
            d_reward.append(r + self.gamma * d_reward[-1])
        return d_reward[::-1]

    def act(self, observation):
        logit, _ = self.net(torch.tensor(observation, dtype=torch.float32))
        prob = torch.softmax(logit, dim=-1)
        action = torch.multinomial(prob, 1).item()
        return action

    def play_episode(self):
        observation = self.environment.reset()
        observations, actions, rewards = [observation], [], []
        done = False
        while not done:
            action = self.act(observation)
            observation, reward, done, _ = self.environment.step(action)
            observations.append(observation)
            actions.append(action)
            rewards.append(reward)
        return observations[:-1], actions, rewards

    def train_step(self):
        observations, actions, rewards = self.play_episode()
        d_rewards = self.discounted_reward(rewards)

        observations = torch.tensor(observations, dtype=torch.float32)
        d_rewards = torch.tensor(d_rewards, dtype=torch.float32)
        logit, _ = self.net(observations)
        p = torch.softmax(logit, dim=-1)
        log_p = torch.log_softmax(logit, dim=-1)
        batch_size = observations.size(0)
        log_p_for_actions = log_p[torch.arange(batch_size), actions]

        policy_loss = (log_p_for_actions * d_rewards).mean()
        entropy = -(p * log_p).sum(-1).mean()
        loss = -policy_loss - self.entropy_reg * entropy
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return policy_loss.item(), entropy.item(), sum(rewards)

    def train(self, n_episodes):
        """train agent for n full episodes"""
        for episode in trange(n_episodes):
            policy_loss, entropy, episode_reward = self.train_step()
            self.writer.add_scalar('policy_loss', policy_loss, episode)
            self.writer.add_scalar('entropy', entropy, episode)
            self.writer.add_scalar('episode_reward', episode_reward, episode)
