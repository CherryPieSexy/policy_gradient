# shamelessly stolen from https://github.com/higgsfield/RL-Adventure/blob/master/common/wrappers.py

import cv2
import gym
from gym import spaces
from collections import deque
import numpy as np

__all__ = ['atari_list', 'make_atari']

atari_list = ['AirRaid-v0', 'Alien-v0', 'Amidar-v0', 'Assault-v0', 'Asterix-v0', 'Asteroids-v0', 'Atlantis-v0',
              'BankHeist-v0', 'BattleZone-v0', 'BeamRider-v0', 'Berzerk-v0', 'Bowling-v0', 'Boxing-v0',
              'Breakout-v0',
              'Carnival-v0', 'Centipede-v0', 'ChopperCommand-v0', 'CrazyClimber-v0',
              'DemonAttack-v0', 'DoubleDunk-v0',
              'ElevatorAction-v0', 'Enduro-v0', 'FishingDerby-v0', 'Freeway-v0', 'Frostbite-v0',
              'KungFuMaster-v0']


class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        lives = self.env.unwrapped.ale.lives()
        if 0 < lives < self.lives:
            done = True
        self.lives = lives
        return obs, reward, done, info

    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs


class ClipReward(gym.RewardWrapper):
    def __init__(self, env):
        super(ClipReward, self).__init__(env)

    def reward(self, reward):
        return np.sign(reward)


class WrapFrame(gym.ObservationWrapper):
    def __init__(self, env):
        """Warp frames to 84x84 as done in the Nature paper and later work."""
        gym.ObservationWrapper.__init__(self, env)
        self.width = 84
        self.height = 84
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.height, self.width, 1), dtype=np.float32)

    def observation(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return frame[:, :, None]


class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        """Stack k last frames"""
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(shp[0], shp[1], shp[2] * k), dtype=np.float32)

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        observation = np.array(list(self.frames), dtype=np.float32).squeeze(-1)
        return (observation / 127.5) - 1.0


def make_atari(env):
    env = gym.make(env)
    env = EpisodicLifeEnv(env)
    env = ClipReward(env)
    env = WrapFrame(env)
    env = FrameStack(env, 4)
    return env
