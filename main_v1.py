import gymnasium as gym
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from env.creditScoring import creditScoring_v1
from model.principals import Principal_v1

def main():
    # hyperparameters
    # 信息：当前的超参数设置旨在快速训练一个合适的代理。如果想收敛到最优策略，请尝试将学习率提高n_episodes10 倍并降低学习率（例如降低至 0.001）。
    learning_rate = 0.001
    n_episodes = 100000
    start_epsilon = 1.0
    epsilon_decay = start_epsilon / (n_episodes / 2)  # reduce the exploration over time
    final_epsilon = 0.1

    env = gym.make("creditScoring_v1", mode='train')
    env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)

    agent = Principal_v1(
        env=env,
        learning_rate=learning_rate,
        initial_epsilon=start_epsilon,
        epsilon_decay=epsilon_decay,
        final_epsilon=final_epsilon,
    )

    for episode in tqdm(range(n_episodes)):
        obs, info = env.reset()
        done = False

        # play one episode
        while not done:
            action = agent.get_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)

            # update the agent:
            # q value is updated; training error is recorded
            agent.update(obs, action, reward, terminated, next_obs)

            # update if the environment is done and the current obs
            # "truncated" does not acctually happen in Blackjack
            done = terminated or truncated
            obs = next_obs

        agent.decay_epsilon()

    return agent, env

def get_moving_avgs(arr, window, convolution_mode):
    return np.convolve(
        np.array(arr).flatten(),
        np.ones(window),
        mode=convolution_mode
    ) / window

def plot_results(agent, env):
    # Smooth over a 500 episode window
    rolling_length = 1000
    fig, axs = plt.subplots(ncols=3, figsize=(12, 5))

    axs[0].set_title("Episode rewards")
    reward_moving_average = get_moving_avgs(
        env.return_queue,
        rolling_length,
        "valid"
    )
    axs[0].plot(range(len(reward_moving_average)), reward_moving_average)

    axs[1].set_title("Episode lengths")
    length_moving_average = get_moving_avgs(
        env.length_queue,
        rolling_length,
        "valid"
    )
    axs[1].plot(range(len(length_moving_average)), length_moving_average)

    axs[2].set_title("Training Error")
    training_error_moving_average = get_moving_avgs(
        agent.training_error,
        rolling_length,
        "same"
    )
    axs[2].plot(range(len(training_error_moving_average)), training_error_moving_average)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    agent, env = main()
    plot_results(agent, env)

    