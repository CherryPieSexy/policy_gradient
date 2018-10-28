from time import sleep


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
