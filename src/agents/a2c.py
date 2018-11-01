from tqdm import trange
import torch
import torch.nn.functional as F


class A2C:
    agent = 'Actor-Critic'
    """Actor-Critic agent"""
    def __init__(self, env_pool,
                 net, device, optimizer, entropy_reg, writer,
                 rollout_len, normalize_advantage, gamma=0.99):
        self.env_pool = env_pool
        self.net = net
        self.device = device
        self.optimizer = optimizer
        self.entropy_reg = entropy_reg
        self.writer = writer
        self.rollout_len = rollout_len
        self.normalize_advantage = normalize_advantage
        self.gamma = gamma

    def act(self, observation):
        with torch.no_grad():
            logit, _ = self.net(torch.tensor(observation, dtype=torch.float32))

        prob = torch.softmax(logit, dim=-1)
        action = torch.multinomial(prob, 1)
        return action.squeeze().cpu().numpy().tolist()

    def collect_batch(self, observation):
        """plays a game for n_steps with current policy, returns collected experience in form [Time, Batch]"""
        observations, actions, rewards, dones = [observation], [], [], []
        for _ in range(self.rollout_len):
            action = self.act(observation)
            observation, reward, done = self.env_pool.step(action)
            observations.append(observation)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
        return observations, actions, rewards, dones

    def train_on_batch(self, observations, actions, rewards, dones):
        observations = torch.tensor(observations, dtype=torch.float32, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)

        time, batch = rewards.size()
        obs_size = observations.size()[2:]

        logits, values = self.net(observations[:-1].view(time*batch, *obs_size))
        p = F.softmax(logits, dim=-1)
        log_p = F.log_softmax(logits, dim=-1)
        log_p_for_actions = log_p[torch.arange(time*batch), actions.view(time*batch)].view(time, batch)
        with torch.no_grad():
            _, next_values = self.net(observations[1:].view(time*batch, *obs_size))
        rewards = rewards.view(time*batch)
        dones = dones.view(time*batch)
        # one-step advantage estimation. TODO: change to GAE
        advantage = rewards + self.gamma * (1.0 - dones) * next_values[:, 0] - values[:, 0]
        advantage = advantage.view(time, batch)

        if self.normalize_advantage:
            normalized_advantage = (advantage - advantage.mean()) / (advantage.std())
            policy_loss = (log_p_for_actions * normalized_advantage.detach()).mean()
        else:
            policy_loss = (log_p_for_actions * advantage.detach()).mean()
        value_loss = (advantage ** 2).mean()
        entropy = -(p * log_p).sum(-1).mean()
        loss = value_loss - policy_loss - self.entropy_reg * entropy
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return policy_loss.item(), value_loss.item(), entropy.item()

    def train(self, num_training_steps):
        """train agent for n training steps"""
        observation = self.env_pool.reset()
        for step in trange(num_training_steps):
            observations, actions, rewards, dones = self.collect_batch(observation)
            policy_loss, value_loss, entropy = self.train_on_batch(observations, actions, rewards, dones)
            self.writer.add_scalar('policy_loss', policy_loss, step)
            self.writer.add_scalar('value_loss', value_loss, step)
            self.writer.add_scalar('entropy', entropy, step)
            observation = observations[-1]
