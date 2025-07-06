import gymnasium as gym
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from env.creditScoring import creditScoring_v1
from model.principals import Principal_v1
import csv

# hyperparameters
max_time_steps = 2000 # maximum time steps per episode
rolling_length = max_time_steps//20  # for plotting moving averages
init_cost_pram=2.0,
learning_rate = 1e-5
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
    fig, axs = plt.subplots(ncols=3, figsize=(12, 5))

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

    # Plot Training Rewards with Moving Average
    axs[2].set_title(f"Training rewards (Smoothed over {rolling_length} steps)")
    acc_moving_average = get_moving_avgs(
        agent.training_rewards,
        rolling_length,
        "valid"
    )
    axs[2].plot(range(len(acc_moving_average)), acc_moving_average, label="Smoothed Rewards")
    axs[2].legend()
    axs[2].set_xlabel("Training Step")
    axs[2].set_ylabel("Reward")

    plt.tight_layout()
    plt.savefig('./result/last_experiment/results.png')
    plt.show()

def training_weights_export(agent):
    """
    记录训练过程中策略权重的变化
    """
    path = './result/last_experiment/training_policy_weights.csv'
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        for weight in agent.training_policy_weights:
            writer.writerow(weight)
    print(f"Training policy weights saved to {path}")

def plot_policy_weights_export(agent):
    weights_array = np.array(agent.training_policy_weights)  # shape: (n_steps, 11)
    n_steps, n_dims = weights_array.shape

    plt.figure(figsize=(12, 6))
    for i in range(n_dims):
        plt.plot(range(n_steps), weights_array[:, i], label=f"Dim {i}")

    plt.title("Policy Weight Evolution During Training")
    plt.xlabel("Training Steps")
    plt.ylabel("Weight Value")
    plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('./result/last_experiment/policy_weights_all_dims.png')
    plt.show()

def training_accuracy_export(agent):
    # 记录训练过程中准确率的变化
    path = './result/last_experiment/training_accuracy.csv'
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        for acc in agent.training_accuracy:
            writer.writerow([acc])
    print(f"Training accuracy details saved to {path}")

    # 记录训练过程中每一步的准确率详情
    path = './result/last_experiment/training_acc_detail.csv'
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['action', 'true_label', 'predicted_label', 'accuracy'])
        for detail in agent.training_acc_detail:
            writer.writerow([
                detail['action'],
                detail['true_label'],
                detail['predicted_label'],
                detail['accuracy']
            ])
    print(f"Training accuracy details saved to {path}")

def testing_accuracy_export(agent):
    # 记录test过程中准确率的变化
    path = './result/last_experiment/testing_accuracy.csv'
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        for acc in agent.testing_accuracy:
            writer.writerow([acc])
    print(f"Testing accuracy details saved to {path}")

    # 记录test过程中每一步的准确率详情
    path = './result/last_experiment/testing_acc_detail.csv'
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['action', 'true_label'])
        for detail in agent.testing_acc_detail:
            writer.writerow([
                detail['action'],
                detail['true_label']
            ])
    print(f"Testing accuracy details saved to {path}")

if __name__ == "__main__":
    print("Current setting:", "normalized data + strategic response.")
    agent, env = main()
    training_weights_export(agent)
    training_accuracy_export(agent)
    testing_accuracy_export(agent)
    plot_policy_weights_export(agent)

    # Plot the results
    plot_results(agent, env, rolling_length=rolling_length)