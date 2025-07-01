import gymnasium as gym
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from env.creditScoring import creditScoring_v1
from model.principals import Principal_v1

# hyperparameters
rolling_length = 500  # for plotting moving averages
max_time_steps = 10000 # maximum time steps per episode
init_cost_pram=2.0,
learning_rate = 0.0001
n_episodes = 1

def main():
    env = gym.make("creditScoring_v1", mode='train')
    # env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)
    env_test = gym.make("creditScoring_v1", mode='test')
    # env_test = gym.wrappers.RecordEpisodeStatistics(env_test, buffer_length=n_episodes)

    agent = Principal_v1(
        env=env,
        learning_rate_actor=learning_rate,
        learning_rate_critic=learning_rate,
        learning_rate_cost=learning_rate,
        init_cost_pram=init_cost_pram,
    )

    for episode in tqdm(range(n_episodes)):
        obs, info = env.reset()
        done = False

        # while not done:
        for step in tqdm(range(max_time_steps), desc=f"Train: Step in episode {episode}"):
            if done:
                break

            action = agent.get_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)

            # 使用 info (true_label) 来更新模型
            agent.update(obs, action, reward, terminated, info)
            env.policy_weight = agent.previous_policy_weight  # update the policy weight in the environment

            # update if the environment is done and the current obs
            done = terminated or truncated
            obs, info = next_obs, info

        # test the model
        obs, info = env_test.reset()
        done = False
        for step in tqdm(range(max_time_steps), desc=f"Test: Step in episode {episode}"):
            if done:
                break
            action = agent.get_action(obs)
            next_obs, reward, terminated, truncated, info = env_test.step(action)
            agent.test_result_record( action, info)

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

def plot_results(agent, env, rolling_length=rolling_length):
    fig, axs = plt.subplots(ncols=2, figsize=(12, 5))

    # Plot Training Accuracy with Moving Average
    axs[0].set_title(f"Training Accuracy (Smoothed over {rolling_length} steps)")
    acc_moving_average = get_moving_avgs(
        agent.training_accuracy,
        rolling_length,
        "valid"
    )
    axs[0].plot(range(len(acc_moving_average)), acc_moving_average, label="Smoothed Accuracy")
    axs[0].legend()
    axs[0].set_xlabel("Training Step")
    axs[0].set_ylabel("Accuracy")
    axs[0].set_ylim(0.0, 1.0) 

    # Plot Testing Accuracy with Moving Average
    axs[1].set_title(f"Testting Accuracy (Smoothed over {rolling_length} steps)")
    acc_moving_average = get_moving_avgs(
        agent.testing_accuracy,
        rolling_length,
        "valid"
    )
    axs[1].plot(range(len(acc_moving_average)), acc_moving_average, label="Smoothed Accuracy")
    axs[1].legend()
    axs[1].set_xlabel("Testing Step")
    axs[1].set_ylabel("Accuracy")
    axs[1].set_ylim(0.0, 1.0) 

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    agent, env = main()
    plot_results(agent, env, rolling_length=rolling_length)
    

    