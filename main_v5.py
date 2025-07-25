import gymnasium as gym
import numpy as np
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt
from env.creditScoring_v5 import creditScoring_v5
from model.principal_v5 import Principal_v5
import csv
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from utils.data_prep_for2D import load_data
import os
import types
import torch


# hyperparameters
max_training_time_steps = 100000
max_testing_time_steps = 100000
n_episodes = 1  # 30
train_rolling_length = max_training_time_steps//200*n_episodes # for plotting moving averages
test_rolling_length = max_testing_time_steps//200*n_episodes
learning_rate = 1e-2
seed = 0 # 0 or 2

# mode = "normalized data + non-strategic response"

def main():
    env = gym.make("creditScoring_v5")
    agent = Principal_v5(
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

            prob, action = agent.get_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            # 使用 info (true_label) 来更新模型
            agent.update(obs, action, reward, terminated, info, prob)
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
        header="w1,w2,w3",
        comments=""
    )

    # test
    env_test = gym.make("creditScoring_v5")
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

    # Plot Training Batch Expected Accuracy
    axs[0].set_title("Training Batch Expected Accuracy per Episode")

    # Approach 1: Use batch accuracy values recorded in agent.training_batch_acc
    # Use batch accuracy values recorded in agent.training_batch_acc
    batch_acc = agent.training_batch_acc
    acc_moving_average = get_moving_avgs(
         batch_acc,
        train_rolling_length//128,  # 128 is the batch size
        "valid"
    )
    axs[0].plot(range(len(acc_moving_average)), acc_moving_average, label="Batch Expected Accuracy")
    # Optional: horizontal line for target accuracy
    # axs[0].axhline(0.7, linestyle='--', label='Target ≈0.7')
    axs[0].legend()
    axs[0].set_xlabel("Episode")
    axs[0].set_ylabel("Expected Accuracy")
    axs[0].set_ylim(0.0, 1.0)
    
    # # Appoach 2: calculated from reward
    # smooth_r = get_moving_avgs(agent.training_rewards, 200_000, "valid")
    # axs[0].plot((smooth_r + 1)/2, label="Expected acc from reward")

    # axs[0].legend()
    # axs[0].set_xlabel("Training Step")
    # axs[0].set_ylabel("Accuracy")
    # # axs[0].set_ylim(0.0, 1.0) 

    # Plot Training Rewards with Moving Average
    axs[1].set_title(f"Training rewards (Smoothed over {test_rolling_length} steps)")
    reawrd_moving_average = get_moving_avgs(
        agent.training_rewards,
        test_rolling_length,
        "valid"
    )
    axs[1].plot(range(len(reawrd_moving_average)), reawrd_moving_average, label="Smoothed Rewards")
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
                detail['expected_accuracy'],
                detail['reward']
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
    # plt.axhline(0.7, color="gray", linestyle="--", label="Target ≈0.7")
    plt.xlabel("Batch #")
    plt.ylabel("Expected Accuracy")
    plt.legend()
    plt.title("Cumulative Expected Accuracy over Batches")
    plt.savefig('./result/last_experiment/batch_expected_acc.png')
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

# ---------- 2D Toy Data Visualization tool functions ----------
import numpy as np
import matplotlib.pyplot as plt
import torch

def visualise_data2D(X, y, save_path=None):
    """
    原始 2D 数据散点图
    正样本蓝色，负样本橙色
    """
    plt.figure()
    plt.scatter(X[y==1,0], X[y==1,1], label='Negative', alpha=0.7)
    plt.scatter(X[y==0,0], X[y==0,1], label='Positive', alpha=0.7)
    plt.legend()
    plt.title("2D Toy Data")
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()

def visualise_separator2D_old(model, X, y, save_path=None):
    """
    在 2D 数据上绘制 model 的线性决策边界和数据点
    要求 model.fc.weight, model.fc.bias 可用
    """
    w = model.fc.weight.detach().cpu().numpy()[0]  # [w0, w1]
    b = model.fc.bias.item()
    # 决策线: w0*x + w1*y + b = 0 -> y = -(w0*x + b)/w1
    xs = np.linspace(X[:,0].min(), X[:,0].max(), 200)
    ys = -(w[0]*xs + b) / (w[1] if abs(w[1])>1e-6 else 1e-6)
    plt.figure()
    plt.scatter(X[y==1,0], X[y==1,1], label='Negative', alpha=0.7)
    plt.scatter(X[y==0,0], X[y==0,1], label='Positive', alpha=0.7)
    plt.plot(xs, ys, 'k--', label='Boundary')
    plt.legend()
    plt.title("Decision Boundary")
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()

def visualize_separator2D(model, X, y, save_path=None):
    # 确保 X 是 torch.Tensor
    if isinstance(X, np.ndarray):
        X = torch.tensor(X, dtype=torch.float32)
    
    if X.shape[1] == 3:
        X = X[:, :-1]  # 只保留前两维用于绘图

    if not X.size(1) == 2:
        return

    x_high, z_high = torch.max(X,0).values.tolist()
    x_low, z_low = torch.min(X,0).values.tolist()

    W = model.fc.weight[0]
    b = model.fc.bias
    
    Xpos = X[y == 0]
    Xneg = X[y == 1]

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.set_xlim([x_low, x_high])
    ax.set_ylim([z_low, z_high])

    ax.scatter(Xpos[:, 0], Xpos[:, 1], marker='+', color='blue')
    ax.scatter(Xneg[:, 0], Xneg[:, 1], marker='_', color='red')

    range_arr = torch.arange(-5, 5)
    # xx = torch.meshgrid(range_arr)[0]
    xx = torch.meshgrid(range_arr, indexing="ij")[0]

    z = (-W[0] * xx - b) * 1. /W[1]
    ax.plot(xx.detach().numpy(), z.detach().numpy(), alpha=1.0, color='green')

    plt.savefig(save_path, dpi=150) if save_path else plt.show()

    print(f"Decision boundary visualized and saved to {save_path}" if save_path else "Decision boundary visualized.")

# ---------- 2D Toy Data Response Visualization ----------
def visualize_2d_response(agent,
                          data_path: str,
                          seed: int = 0,
                          result_dir: str = './result/last_experiment',
                          use_train: bool = True):
    """
    一次性完成 2D toy 数据的原始样本、response 后样本、
    以及在两者上叠加决策边界的可视化，并保存到 result_dir。

    参数:
      agent       -- 训练完毕的 Principal_v5 实例 (含 previous_policy_weight)
      data_path   -- toy 数据 CSV 路径，供 load_data 加载
      seed        -- 加载数据时的随机种子
      result_dir  -- 保存图像的目标目录
      use_train   -- True: 用 train_x/train_y；False: 用 test_x/test_y
    """
    os.makedirs(result_dir, exist_ok=True)

    # 1. 加载 toy 数据
    train_x, train_y, test_x, test_y = load_data(data_path, seed=seed)
    X, y = (train_x, train_y) if use_train else (test_x, test_y)

    # 2. 画原始点云
    visualise_data2D(X, y,
                    save_path=os.path.join(result_dir, 'toy_raw.png'))

    # 3. 用最终 policy_weight 生成 response 样本
    w = agent.previous_policy_weight
    env = creditScoring_v5()
    env.policy_weight = w
    # 对每个样本调用 strategic_response_Close
    X_strat = np.vstack([
        env.strategic_response_Close(x, env.policy_weight)
        for x in X
    ])

    # 4. 画 response 后的点云
    visualise_data2D(X_strat, y,
                    save_path=os.path.join(result_dir, 'toy_response.png'))

    # 5. 构造一个能被 visualise_separator2D 识别的 “假模型”
    model = types.SimpleNamespace()
    model.fc    = types.SimpleNamespace()
    # 假设 policy_weight 最后一维是 bias
    model.fc.weight = torch.tensor(w[:-1].reshape(1, -1), dtype=torch.float32)
    model.fc.bias   = torch.tensor([w[-1]],               dtype=torch.float32)

    # 6. 在原始 & response 后样本上分别画决策边界
    visualize_separator2D(model, X,      y,
                         save_path=os.path.join(result_dir, 'toy_boundary.png'))
    visualize_separator2D(model, X_strat, y,
                         save_path=os.path.join(result_dir, 'toy_boundary_response.png'))
    
    print(f"2D toy data visualization saved to {result_dir}")
    
if __name__ == "__main__":
    # print("Current setting:", mode)
    agent, env = main()
    # training_weights_export(agent)
    training_accuracy_export(agent)
    testing_accuracy_export(agent)
    # training_weights_single_update(agent)

    # Plot the results
    # training_batch_acc(agent)
    plot_policy_weights_export(agent)
    plot_test_auc(agent)
    plot_results_accAndRewards_export(agent, env, train_rolling_length, test_rolling_length)

    visualize_2d_response(
        agent,
        data_path = "./data/generated_2D_data.csv",
        seed      = 0,
        result_dir= "./result/last_experiment",
        use_train = True
    )

