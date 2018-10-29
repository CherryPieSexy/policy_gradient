import torch
from time import sleep


__all__ = ['atari_list', 'play_episode', 'save', 'load']


atari_list = ['AiRraid-v0', 'Alien-v0', 'Amidar-v0', 'Assault-v0', 'Asterix-v0', 'Asteroids-v0', 'Atlantis-v0',
              'BankHeist-v0', 'BattleZone-v0', 'BeamRider-v0', 'Berzerk-v0', 'Bowling-v0', 'Boxing-v0', 'Breakout-v0',
              'Carnival-v0', 'Centipede-v0', 'ChopperCommand-v0', 'CrazyClimber-v0',
              'DemonAttack-v0', 'DoubleDunk-v0',
              'ElevatorAction-v0', 'Enduro-v0', 'FishingDerby-v0', 'Freeway-v0', 'Frostbite-v0',
              'KungFuMaster-v0']


def play_episode(environment, agent, render):
    episode_reward = 0.0
    observation = environment.reset()
    done = False
    while not done:
        action = agent.act(observation)
        observation, reward, done, _ = environment.step(action)
        if render:
            environment.render()
            sleep(0.01)
        episode_reward += reward
    return episode_reward


def save(agent, filename):
    torch.save({
        'agent': agent.agent,
        'net_type': agent.net.type,
        'net': agent.net.state_dict(),
        'optimizer': agent.optimizer.state_dict(),
    }, filename)


def load(filename, agent):
    checkpoint = torch.load(filename)
    if agent.agent == checkpoint['agent']:
        if agent.net.type == checkpoint['net_type']:
            agent.net.load_state_dict(checkpoint['net'])
            agent.optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            raise BaseException('net type in checkpoint does not correspond to agent\'s net type')
    else:
        raise BaseException('trying to load {} into {} agent'.format(agent.agent, checkpoint['agent']))
