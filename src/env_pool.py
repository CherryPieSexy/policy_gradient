import numpy as np
from collections import deque

__all__ = ['EnvPool']


class EnvPool:
    def __init__(self, make_env, env_name, n_envs, writer):
        self.environments = [make_env(env_name).env for _ in range(n_envs)]
        self.episodes_done = 0
        self.current_rewards = np.zeros(n_envs, dtype=np.float32)
        self.finished_rewards = [deque() for _ in range(n_envs)]
        self.writer = writer

    def reset(self):
        return [env.reset() for env in self.environments]

    def step(self, actions):
        results = [env.step(a) for env, a in zip(self.environments, actions)]
        observation, reward, done, _ = map(list, zip(*results))
        self.current_rewards += np.array(reward, dtype=np.float32)

        for i in range(len(self.environments)):
            if done[i]:
                observation[i] = self.environments[i].reset()
                self.finished_rewards[i].append(self.current_rewards[i])
                self.check_done()
                self.current_rewards[i] = 0

        return observation, reward, done

    def check_done(self):
        """if all environments finished at least one episode - write statistics"""
        full = True
        for env_rewards in self.finished_rewards:
            if len(env_rewards) > 0:
                continue
            else:
                full = False
                break
        if full:
            self.episodes_done += 1
            finished_episode_reward = []
            for env_rewards in self.finished_rewards:
                finished_episode_reward.append(env_rewards.popleft())
            self.writer.add_scalar('mean_worker_reward', np.mean(finished_episode_reward), self.episodes_done)
