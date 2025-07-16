import gymnasium as gym
import numpy as np
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt
from env.creditScoring_v3 import creditScoring_v3
from model.principal_v3 import Principal_v3
import csv
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# hyperparameters
max_training_time_steps = 100000
max_testing_time_steps = 100000
n_episodes = 1 # 300
train_rolling_length = max_training_time_steps//200*n_episodes # for plotting moving averages
test_rolling_length = max_testing_time_steps//200*n_episodes
learning_rate = 1e-2

# mode = "normalized data + non-strategic response"

def main():
    env = gym.make("creditScoring_v3")
    agent = Principal_v3(
        env=env,
        learning_rate_actor=learning_rate,
        learning_rate_critic=learning_rate,
        learning_rate_cost=learning_rate,
    )
    env.policy_weight = agent.previous_policy_weight

    for episode in tqdm(range(n_episodes)):
        # train
        obs, info = env.reset()
        env.policy_weight = agent.previous_policy_weight
        done = False
        for step in tqdm(range(max_training_time_steps), desc=f"Train: Step in episode {episode}"):
            if done:
                break

            _, action = agent.get_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            # 使用 info (true_label) 来更新模型
            agent.update(obs, action, reward, terminated, info)
            env.policy_weight = agent.previous_policy_weight  # update the policy weight in the environment

            # update if the environment is done and the current obs
            done = terminated or truncated
            obs, info = next_obs, info
    print(f"\nTraining: Batch update count: {agent.batch_update_count}\n")

    # save the learned policy weights
    learned_policy_weight = agent.previous_policy_weight
    path = './result/last_experiment/learned_model.csv'
    np.savetxt(
        path,
        learned_policy_weight[np.newaxis, :],  # 把 shape=(11,)→(1,11)
        delimiter=",",
        header="w1,w2,w3,w4,w5,w6,w7,w8,w9,w10,w11",
        comments=""
    )

    # test
    env_test = gym.make("creditScoring_v3")
    env_test.mode = 'test'
    obs, info = env_test.reset()
    env_test.policy_weight = agent.previous_policy_weight
    done = False
    for step in tqdm(range(max_testing_time_steps), desc=f"Test: Step in episode {episode}"):
            if done:
                break
            prob, action = agent.get_action(obs, stochastic=False)
            next_obs, reward, terminated, truncated, info = env_test.step(action)
            agent.test_result_record(action, info, prob)

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

def plot_results_accAndRewards_export(agent, env, train_rolling_length=train_rolling_length, test_rolling_length = test_rolling_length):
    fig, axs = plt.subplots(ncols=2, figsize=(12, 5))

    # Plot Training Accuracy with Moving Average
    axs[0].set_title(f"Training Expected Accuracy (Smoothed over {train_rolling_length} steps)")
    
    # # Approach 1: directly use the training_expected_acc_list
    # # 先扁平化成一个 list of float
    # expected_list = [ e['expected_accuracy'] for e in agent.training_acc_detail ]
    # acc_moving_average = get_moving_avgs(
    #     expected_list,
    #     train_rolling_length,
    #     "valid"
    # )

    # axs[0].plot(range(len(acc_moving_average)), acc_moving_average, label="Smoothed Accuracy")
    
    # Appoach 2: calculated from reward
    smooth_r = get_moving_avgs(agent.training_rewards, 200_000, "valid")
    axs[0].plot((smooth_r + 1)/2, label="Expected acc from reward")

    axs[0].legend()
    axs[0].set_xlabel("Training Step")
    axs[0].set_ylabel("Accuracy")
    # axs[0].set_ylim(0.0, 1.0) 

    # Plot Training Rewards with Moving Average
    axs[1].set_title(f"Training rewards (Smoothed over {test_rolling_length} steps)")
    acc_moving_average = get_moving_avgs(
        agent.training_rewards,
        test_rolling_length,
        "valid"
    )
    axs[1].plot(range(len(acc_moving_average)), acc_moving_average, label="Smoothed Rewards")
    axs[1].legend()
    axs[1].set_xlabel("Training Step")
    axs[1].set_ylabel("Reward")

    # # Plot Testing Accuracy with Moving Average
    # axs[2].set_title(f"Testting Accuracy (Smoothed over {test_rolling_length} steps)")
    # acc_moving_average = get_moving_avgs(
    #     agent.testing_accuracy,
    #     test_rolling_length,
    #     "valid"
    # )
    # axs[2].plot(range(len(acc_moving_average)), acc_moving_average, label="Smoothed Accuracy")
    # axs[2].legend()
    # axs[2].set_xlabel("Testing Step")
    # axs[2].set_ylabel("Accuracy")
    # axs[2].set_ylim(0.0, 1.0)

    plt.tight_layout()
    plt.savefig('./result/last_experiment/results.png')
    print("Results plot saved to './result/last_experiment/results.png'")
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

def training_weights_single_update(agent):
    """
    记录训练过程中策略权重的变化
    """
    path = './result/last_experiment/training_weights_single_update.csv'
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        for change in agent.training_single_policy_weight_update:
            writer.writerow(change)
    print(f"Training policy weight-change saved to {path}")

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
    plt.close()

def training_accuracy_export(agent):
    # # 记录训练过程中准确率的变化
    # path = './result/last_experiment/training_accuracy.csv'
    # with open(path, 'w', newline='') as f:
    #     writer = csv.writer(f)
    #     for acc in agent.training_accuracy:
    #         writer.writerow([acc])
    # print(f"Training accuracy details saved to {path}")

    # 记录训练过程中每一步的准确率详情
    path = './result/last_experiment/training_acc_detail.csv'
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['predicted_prob', 'predicted_label', 'true_label', 'expected_accuracy'])
        for detail in agent.training_acc_detail:
            writer.writerow([
                detail['predicted_prob'],
                detail['predicted_label'],
                detail['true_label'],
                detail['expected_accuracy']
            ])
    print(f"Training accuracy details saved to {path}")

def training_batch_acc(agent):
    # 记录训练过程中 batch acc 的变化
    path = './result/last_experiment/train_batch_acc.csv'
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        for acc in agent.training_batch_acc:
            writer.writerow([acc])
    print(f"Training batch acc saved to {path}")

    batch_acc = agent.training_batch_acc  # List[float], 每个元素对应一批次的 expected accuracy

    # 计算累计平均
    cumavg = np.cumsum(batch_acc) / (np.arange(len(batch_acc)) + 1)

    plt.figure(figsize=(8,4))
    plt.plot(cumavg, label="Cumulative Expected Acc")
    plt.axhline(0.7, color="gray", linestyle="--", label="Target ≈0.7")
    plt.xlabel("Batch #")
    plt.ylabel("Expected Accuracy")
    plt.ylim(0.4, 0.8)
    plt.legend()
    plt.title("Cumulative Expected Accuracy over Batches")
    # plt.show()
    plt.close()
    print("Training batch expected accuracy plot saved to './result/last_experiment/batch_expected_acc.png'")

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

def plot_test_auc(agent):
    """
    使用 self.testing_acc_detail 绘制 ROC 曲线并计算 AUC 值
    要求每个记录中包含 'true_label' 和 'prob'
    """

    # 提取真实标签和预测概率
    y_true = [int(item['true_label']) for item in agent.testing_acc_detail]
    y_score = [item['prob'] for item in agent.testing_acc_detail]

    # 计算 ROC 曲线所需数据
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    # 打印 AUC 值
    print(f"AUC Score: {roc_auc:.4f}")
    
    # 绘制 ROC 曲线
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
    plt.savefig('./result/last_experiment/test_auc.png')
    # plt.show()
    plt.close()

if __name__ == "__main__":
    # print("Current setting:", mode)
    agent, env = main()
    # training_weights_export(agent)
    training_accuracy_export(agent)
    # testing_accuracy_export(agent)
    # training_weights_single_update(agent)

    # Plot the results
    training_batch_acc(agent)
    plot_policy_weights_export(agent)
    plot_test_auc(agent)
    plot_results_accAndRewards_export(agent, env, train_rolling_length, test_rolling_length)