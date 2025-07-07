import gymnasium as gym
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
import torch
import csv

# default hyperparameters
learning_rate = 0.0001
init_cost_pram = 2.0
discount_factor = 0.99
predict_label_threshold = 0.5
pos_buffer_size = 5
pos_neg_ratio = 1

# a principal for the credit scoring v1 environment
class Principal_v1:
    def __init__(
        self,
        env: gym.Env,
        learning_rate_critic: float = learning_rate,
        learning_rate_actor: float = learning_rate,
        learning_rate_cost: float = learning_rate,
        pos_buffer_size = pos_buffer_size,
        pos_neg_ratio = pos_neg_ratio,
        init_cost_pram:  float = 2.0, # same as in the "made practical" paper 
        discount_factor: float = 0.99,
    ):
        """Initialize a Reinforcement Learning agent with an empty dictionary
        of state-action values (q_values), a learning rate and an epsilon.

        Args:
            env: The training environment
            learning_rate: The learning rate
            initial_epsilon: The initial epsilon value
            epsilon_decay: The decay for epsilon in each step
            final_epsilon: The final epsilon value
            discount_factor: The discount factor for computing the Q-value
        """
        self.env = env
        self.discount_factor = discount_factor
        self.lr_a = learning_rate_actor
        # hyperparameter: initial policy weight for the classifier
        self.previous_policy_weight = np.ones(10+1, dtype=np.float64) * 0.01 # policy weight for the classifier sigmoid(w*x + b)
        
        self.lr_c = learning_rate_critic
        # q value weights for the classifier (v*(s, a) + b)
        # +1: bias term,  +1: action term
        self.q_weights = np.ones(10+1+1, dtype=np.float64) * 0.01

        # initial cost parameter estimation for the principal
        self.cost_pram_estimation = np.full(shape=10, fill_value=init_cost_pram, dtype=np.float64)
        self.lr_cost = learning_rate_cost

        # buffer craeted for both pos sample and neg sample
        self.pos_buffer, self.neg_buffer = [], []
        self.pos_buffer_size = pos_buffer_size
        self.neg_buffer_size = int(pos_buffer_size * pos_neg_ratio)

        # record training and testing process
        self.batch_update_count = 0
        self.training_error = []
        self.training_accuracy = []
        self.training_acc_detail = []
        self.training_rewards = []
        self.training_policy_weights = []

        self.testing_accuracy = []
        self.testing_acc_detail = []

    # policy function
    def get_action(self, obs: np.ndarray) -> int:
        """
        making predictions based on the current observation and take strategic response into account.
        
        参数:
            obs (np.ndarray): 观察值 (feature vector)
            
        返回:
            action (int): 0 or 1
        """
        # 添加 bias term 到 obs
        obs_with_bias = np.append(obs, 1.0)  # shape (11,)
        
        # 计算 logits: W^T x + b
        logits = np.dot(self.previous_policy_weight, obs_with_bias)
        
        # 计算概率
        prob = 1 / (1 + np.exp(-logits))
        
        # 阈值判断
        return prob, 1 if prob > predict_label_threshold else 0
        
    def update_old(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        terminated: bool,
        info: dict
    ):
        """
        先放到 buffer 里, buffer 满了然后更新模型:
        使用 QAC 算法更新策略（Actor）和 Q 值函数（Critic）
        参数:
            obs: 当前状态特征
            action: 执行的动作
            reward: 获得的奖励
            terminated: 是否终止
            info: 其他信息（可选）
        """
        true_label = info['true_label']
        if true_label == 1:
            self.pos_buffer.append((obs, action, reward, terminated))
        else:
            self.neg_buffer.append((obs, action, reward, terminated))

        # Step 1: append bias to obs
        obs_with_bias = np.append(obs, 1.0)  # shape: (11,)
        
        # Step 2: construct the input of Q function: [obs_with_bias, action]
        q_input = np.append(obs_with_bias, action)  # shape: (12,)
        
        # Step 3: calculate the predicted Q value
        q_value = np.dot(self.q_weights, q_input)

        # Step 4: calculate the maximum Q value for the next state (for calculating TD target)
        if not terminated:
            next_obs = info.get("next_obs", None)
            if next_obs is not None:
                # 2 actions: 0 or 1
                q_next_0 = np.dot(self.q_weights, np.append(np.append(next_obs, 1.0), 0))
                q_next_1 = np.dot(self.q_weights, np.append(np.append(next_obs, 1.0), 1))
                max_q_next = max(q_next_0, q_next_1)
            else:
                max_q_next = 0.0
        else:
            max_q_next = 0.0

        # Step 5: TD Target & TD Error
        td_target = reward + self.discount_factor * max_q_next
        td_error = td_target - q_value

        # Step 6:（Critic）update Q function weights 
        # (times q_input since it is a linear function, the gradient is the input)
        self.q_weights += self.lr_c * td_error * q_input

        # Step 7: （Actor）calculate the gradient of the log policy
        logits = np.dot(self.previous_policy_weight, obs_with_bias)
        prob = 1 / (1 + np.exp(-logits))  # sigmoid
        grad_log_pi = (action - prob) * obs_with_bias # the gradient of log policy function log(sigmoid(w*x + b))is: (action - prob) * obs_with_bias

        # Step 8: (Actor) using TD-error as advantage to update the policy weight
        grad_log_pi = np.clip(grad_log_pi, -1.0, 1.0)  # clip the gradient to avoid large updates
        self.previous_policy_weight += self.lr_a * td_error * grad_log_pi

        # Record training results
        self.training_error.append(abs(prob - info['true_label']))
        self.training_rewards.append(reward)
        self.training_policy_weights.append(self.previous_policy_weight.copy())

        pred = self.get_action(obs)[1]
        accuracy = 1.0 if pred == int(info['true_label']) else 0.0
        self.training_accuracy.append(accuracy)
        self.training_acc_detail.append({
            'action': action,
            'true_label': info['true_label'],
            'predicted_label': pred,
            'accuracy': accuracy
        })

    def batch_update(self):
        if len(self.pos_buffer) < self.pos_buffer_size or len(self.neg_buffer) < self.neg_buffer_size:
            return  # 不足够数据，不更新

        self.batch_update_count += 1
        # 合并并打乱
        batch = self.pos_buffer + self.neg_buffer
        np.random.shuffle(batch)

        for obs, action, reward, terminated, next_obs in batch:
            # Critic部分
            obs_with_bias = np.append(obs, 1.0)                # shape (11,)
            q_input = np.append(obs_with_bias, action)         # shape (12,)
            q_value = np.dot(self.q_weights, q_input)

            # 处理 next_obs
            if not terminated and next_obs is not None:
                next_obs_with_bias = np.append(next_obs, 1.0)
                q_next_0 = np.dot(self.q_weights, np.append(next_obs_with_bias, 0))
                q_next_1 = np.dot(self.q_weights, np.append(next_obs_with_bias, 1))
                max_q_next = max(q_next_0, q_next_1)
            else:
                max_q_next = 0.0

            td_target = reward + self.discount_factor * max_q_next
            td_error = td_target - q_value

            # Critic 更新
            self.q_weights += self.lr_c * td_error * q_input

            # Actor 更新
            logits = np.dot(self.previous_policy_weight, obs_with_bias)
            prob = 1 / (1 + np.exp(-logits))
            grad_log_pi = (action - prob) * obs_with_bias
            grad_log_pi = np.clip(grad_log_pi, -1.0, 1.0)
            self.previous_policy_weight += self.lr_a * td_error * grad_log_pi

        # 清空 buffer
        self.pos_buffer.clear()
        self.neg_buffer.clear()

    def update(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        terminated: bool,
        info: dict
    ):
        true_label = info['true_label']
        next_obs = info.get('next_obs', None)
        sample = (obs, action, reward, terminated, next_obs)

        if true_label == 1:
            if len(self.pos_buffer) < self.pos_buffer_size:
                self.pos_buffer.append(sample)
            else:
                # 正样本已满 -> 随机替换
                idx = np.random.randint(self.pos_buffer_size)
                self.pos_buffer[idx] = sample
        else:
            if len(self.neg_buffer) < self.neg_buffer_size:
                self.neg_buffer.append(sample)
            else:
                # 负样本已满 -> 随机替换
                idx = np.random.randint(self.neg_buffer_size)
                self.neg_buffer[idx] = sample

        # 满足触发条件时才 batch 更新
        if len(self.pos_buffer) >= self.pos_buffer_size and len(self.neg_buffer) >= self.neg_buffer_size:
            self.batch_update()

        # 记录指标
        prob, pred = self.get_action(obs)
        self.training_error.append(abs(prob - info['true_label']))
        self.training_rewards.append(reward)
        self.training_policy_weights.append(self.previous_policy_weight.copy())
        accuracy = 1.0 if pred == int(info['true_label']) else 0.0
        self.training_accuracy.append(accuracy)
        self.training_acc_detail.append({
            'predicted_prob': prob,
            'predicted_label': action,
            'true_label': info['true_label'],
            'accuracy': accuracy
        })


    def test_result_record(self, action: int, info: dict, prob: float):
        """
        记录测试结果
        
        参数:
            action: 执行的动作
            info: 包含 true_label 的字典
        """
        accuracy = 1.0 if action == int(info['true_label']) else 0.0
        self.testing_accuracy.append(accuracy)

        # detail record
        self.testing_acc_detail.append({
            'action': action, 
            'true_label': info['true_label'],
            'prob': prob
            })