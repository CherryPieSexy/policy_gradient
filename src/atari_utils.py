# TODO:
# observation wrapper: clip image to 1x84x84
# stack 4 images into one 4x84x84
# return done==True every episode end
import gym
import numpy as np


class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        super(EpisodicLifeEnv, self).__init__(env)
        pass

    def step(self, action):
        pass

    def reset(self, **kwargs):
        pass


class ClipReward(gym.RewardWrapper):
    def __init__(self, env):
        super(ClipReward, self).__init__(env)

    def reward(self, reward):
        return np.sign(reward)


class WrapFrame(gym.ObservationWrapper):
    pass


class FrameStack(gym.Wrapper):
    pass


class ImageToPyTorch(gym.ObservationWrapper):
    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)


def make_atari(env):
    env = gym.make(env)
    env = EpisodicLifeEnv(env)
    # env = ClipReward(env)
    # env = WrapFrame(env)
    # env = FrameStack(env)
    return env
