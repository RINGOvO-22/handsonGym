import gymnasium as gym
import numpy as np
import csv
import os
from tqdm import tqdm
from model.principal_v2_PPO import Principal_v2_PPO
from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
from env.creditScoring import creditScoring_v1

def get_moving_avgs(arr, window, convolution_mode):
    return np.convolve(
        np.array(arr).flatten(),
        np.ones(window),
        mode=convolution_mode
    ) / window


def plot_results_accAndRewards_export(agent, train_rolling_length, test_rolling_length):
    fig, axs = plt.subplots(ncols=3, figsize=(12, 5))

    axs[0].set_title(f"Training Accuracy (Smoothed over {train_rolling_length} steps)")
    acc_moving_average = get_moving_avgs(agent.training_accuracy, train_rolling_length, "valid")
    axs[0].plot(range(len(acc_moving_average)), acc_moving_average, label="Smoothed Accuracy")
    axs[0].legend()
    axs[0].set_xlabel("Training Step")
    axs[0].set_ylabel("Accuracy")
    axs[0].set_ylim(0.0, 1.0)

    axs[1].set_title(f"Training rewards (Smoothed over {test_rolling_length} steps)")
    acc_moving_average = get_moving_avgs(agent.training_rewards, test_rolling_length, "valid")
    axs[1].plot(range(len(acc_moving_average)), acc_moving_average, label="Smoothed Rewards")
    axs[1].legend()
    axs[1].set_xlabel("Training Step")
    axs[1].set_ylabel("Reward")
    
    axs[2].set_title(f"Testing Accuracy (Smoothed over {test_rolling_length} steps)")
    acc_moving_average = get_moving_avgs(agent.testing_accuracy, test_rolling_length, "valid")
    axs[2].plot(range(len(acc_moving_average)), acc_moving_average, label="Smoothed Accuracy")
    axs[2].legend()
    axs[2].set_xlabel("Testing Step")
    axs[2].set_ylabel("Accuracy")
    axs[2].set_ylim(0.0, 1.0)

    

    plt.tight_layout()
    plt.savefig('./result/last_experiment_v2/results.png')
    plt.show()


def plot_policy_weights_export(agent):
    weights_array = np.array(agent.training_policy_weights)
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
    plt.savefig('./result/last_experiment_v2/policy_weights_all_dims.png')
    plt.show()


def main():
    env = gym.make("creditScoring_v1", mode="train")
    env_test = gym.make("creditScoring_v1", mode="test")

    obs_dim = env.observation_space.shape[0]
    agent = Principal_v2_PPO(obs_dim=obs_dim)

    max_training_time_steps = 149998
    max_testing_time_steps = 100000
    n_episodes = 10
    train_rolling_length = max_training_time_steps // 20 * n_episodes
    test_rolling_length = max_testing_time_steps // 20 * n_episodes

    # train
    for episode in tqdm(range(n_episodes)):
        obs, info = env.reset()
        done = False
        for step in tqdm(range(max_training_time_steps), desc="Training"):
            if done:
                break

            action, log_prob, value, prob = agent.get_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)

            agent.store_transition(
                obs=obs,
                action=action,
                reward=reward,
                done=terminated,
                next_obs=next_obs,
                log_prob=log_prob,
                value=value,
                true_label=info["true_label"]
            )

            done = terminated or truncated
            obs = next_obs if not done else env.reset()[0]

    # test
    obs, info = env_test.reset()
    done = False
    for step in tqdm(range(max_testing_time_steps), desc="Testing"):
        if done:
            break

        action, log_prob, value, prob = agent.get_action(obs)
        next_obs, reward, terminated, truncated, info = env_test.step(action)

        agent.test_result_record(action, info, prob)

        done = terminated or truncated
        obs = next_obs if not done else env_test.reset()[0]

    os.makedirs("./result/last_experiment_v2", exist_ok=True)

    with open("./result/last_experiment_v2/testing_accuracy.csv", mode="w", newline="") as f:
        writer = csv.writer(f)
        for acc in agent.testing_accuracy:
            writer.writerow([acc])

    with open("./result/last_experiment_v2/testing_acc_detail.csv", mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["action", "true_label"])
        for detail in agent.testing_acc_detail:
            writer.writerow([detail['action'], detail['true_label']])

    with open("./result/last_experiment_v2/training_accuracy.csv", mode="w", newline="") as f:
        writer = csv.writer(f)
        for acc in agent.training_accuracy:
            writer.writerow([acc])

    with open("./result/last_experiment_v2/training_acc_detail.csv", mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["predicted_prob", "predicted_label", "true_label", "accuracy"])
        for detail in agent.training_acc_detail:
            writer.writerow([
                detail['predicted_prob'],
                detail['predicted_label'],
                detail['true_label'],
                detail['accuracy']
            ])

    with open("./result/last_experiment_v2/training_rewards.csv", mode="w", newline="") as f:
        writer = csv.writer(f)
        for reward in agent.training_rewards:
            writer.writerow([reward])

    with open("./result/last_experiment_v2/training_policy_weights.csv", mode="w", newline="") as f:
        writer = csv.writer(f)
        for weight in agent.training_policy_weights:
            writer.writerow(weight)

    y_true = [int(item['true_label']) for item in agent.testing_acc_detail]
    y_score = [item['prob'] for item in agent.testing_acc_detail]
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    print(f"AUC Score: {roc_auc:.4f}")

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('./result/last_experiment_v2/test_auc.png')
    plt.show()

    plot_policy_weights_export(agent)
    plot_results_accAndRewards_export(agent, train_rolling_length, test_rolling_length)
    print("Test Accuracy (mean):", np.mean(agent.testing_accuracy))


if __name__ == "__main__":
    main()