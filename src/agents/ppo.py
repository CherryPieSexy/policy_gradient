from tqdm import trange
import torch
import torch.nn.functional as F


class PPO:
    agent = 'Proximal Policy Optimization'
    """Proximal Policy Optimization algorithm"""
    def __init__(self, env_pool,
                 net, device, optimizer, entropy_reg, writer,
                 rollout_len, normalize_advantage,
                 ppo_batch_size, num_ppo_epochs, ppo_eps, gae_lambda,
                 gamma=0.99):
        self.env_pool = env_pool
        self.net = net
        self.device = device
        self.optimizer = optimizer
        self.entropy_reg = entropy_reg
        self.writer = writer
        self.rollout_len = rollout_len
        self.normalize_advantage = normalize_advantage

        self.ppo_batch_size = ppo_batch_size
        self.num_ppo_epochs = num_ppo_epochs
        self.ppo_eps = ppo_eps
        self.gae_lambda = gae_lambda

        self.gamma = gamma
        print(ppo_batch_size, num_ppo_epochs, ppo_eps, gae_lambda)

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

    def compute_gae(self, rewards, values, next_values):
        step_gae = 0
        gae = []
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * next_values[step] - values[step]
            step_gae = delta + self.gamma * self.gae_lambda * step_gae
            gae.append(step_gae)
        return torch.stack(gae[::-1], dim=0)

    def train_on_batch(self, observations, actions, rewards, dones):
        observations = torch.tensor(observations, dtype=torch.float32, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)

        time, batch = rewards.size()
        obs_size = observations.size()[2:]

        # pi_old logits and values
        with torch.no_grad():
            logits, values = self.net(observations[:-1].view(time*batch, *obs_size))
        log_p = F.log_softmax(logits, dim=-1)
        log_p_for_actions = log_p[torch.arange(time*batch), actions.view(time*batch)].view(time, batch)
        with torch.no_grad():
            _, next_values = self.net(observations[1:].view(time*batch, *obs_size))
        advantage = self.compute_gae(rewards, values.view(time, batch), (1.0 - dones) * next_values.view(time, batch))

        # if self.normalize_advantage:
        #     advantage = (advantage - advantage.mean()) / (advantage.std())
        epoch_policy_loss, epoch_entropy, epoch_value_loss = 0., 0., 0.
        for _ in range(self.num_ppo_epochs):
            indices = torch.randint(0, batch, size=(self.ppo_batch_size,), dtype=torch.long).cpu().numpy().tolist()
            policy_loss, value_loss, entropy = self.ppo_epoch(observations[:-1, indices].view(-1, *obs_size),
                                                              actions[:, indices].view(-1),
                                                              rewards[:, indices].view(-1),
                                                              dones[:, indices].view(-1),
                                                              observations[1:, indices].view(-1, *obs_size),
                                                              advantage[:, indices].view(-1).detach(),
                                                              log_p_for_actions[:, indices].view(-1))
            epoch_value_loss += value_loss
            epoch_policy_loss += policy_loss
            epoch_entropy += entropy
        return epoch_policy_loss / self.num_ppo_epochs, \
               epoch_value_loss / self.num_ppo_epochs, \
               epoch_entropy / self.num_ppo_epochs

    def ppo_epoch(self, observations, actions, rewards, dones,
                  next_observations, advantages, old_log_p_for_a):
        """Computes ppo losses. Accepts only flat parameters"""
        logit, value = self.net(observations)
        p = F.softmax(logit, dim=-1)
        log_p = F.log_softmax(logit, dim=-1)
        log_p_for_a = log_p[torch.arange(self.rollout_len * self.ppo_batch_size), actions]

        ratio = (log_p_for_a - old_log_p_for_a).exp()
        surrogate1 = ratio * advantages
        surrogate2 = torch.clamp(ratio, 1.0-self.ppo_eps, 1.0+self.ppo_eps) * advantages
        policy_loss = torch.min(surrogate1, surrogate2).mean()

        with torch.no_grad():
            _, next_values = self.net(next_observations)

        target_value = rewards + self.gamma * (1.0 - dones) * next_values[:, 0]
        value_loss = ((value[:, 0] - target_value) ** 2).mean()
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
