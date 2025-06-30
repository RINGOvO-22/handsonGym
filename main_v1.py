import gymnasium as gym
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from env.creditScoring import creditScoring_v1
from model.principals import Principal_v1

def main():
    # hyperparameters
    learning_rate = 0.001
    n_episodes = 10

    env = gym.make("creditScoring_v1", mode='train')
    env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)

    agent = Principal_v1(
        env=env,
        learning_rate_actor=learning_rate,
        learning_rate_critic=learning_rate,
        learning_rate_cost=learning_rate,
        init_cost_pram=2.0,
    )

    for episode in tqdm(range(n_episodes)):
        obs, info = env.reset()
        done = False

        # play one episode
        while not done:
            action = agent.get_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)

            # 使用 info (true_label) 来更新模型（也可以使用 reward，看你的设计）
            agent.update(obs, action, reward, terminated, info)
            env.policy_weight = agent.previous_policy_weight

            # update if the environment is done and the current obs
            done = terminated or truncated
            obs, info = next_obs, info

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

    