import gymnasium as gym
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
import torch

# a principal for the credit scoring v1 environment
class Principal_v1:
    def __init__(
        self,
        env: gym.Env,
        learning_rate_critic: float = 0.001,
        learning_rate_actor: float = 0.001,
        learning_rate_cost: float = 0.001,
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
        self.previous_policy_weight = np.zeros(10+1, dtype=np.float32) # policy weight for the classifier sigmoid(w*x + b)
        
        self.lr_c = learning_rate_critic
        # q value weights for the classifier (v*(s, a) + b)
        # +1: bias term,  +1: action term
        self.q_weights = np.ones(10+1+1, dtype=np.float32) * 0.1

        # initial cost parameter estimation for the principal
        self.cost_pram_estimation = np.full(shape=10, fill_value=init_cost_pram, dtype=np.float32)
        self.lr_cost = learning_rate_cost

        self.training_error = []
        self.training_accuracy = []
        self.training_rewards = []
        self.testing_accuracy = []

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
        return 1 if prob > 0.5 else 0
        
    def update(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        terminated: bool,
        info: dict
    ):
        """
        使用 QAC 算法更新策略（Actor）和 Q 值函数（Critic）
        
        参数:
            obs: 当前状态特征
            action: 执行的动作
            reward: 获得的奖励
            terminated: 是否终止
            info: 其他信息（可选）
        """

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
        pred = 1 if prob > 0.5 else 0
        accuracy = 1.0 if pred == int(info['true_label']) else 0.0
        self.training_accuracy.append(accuracy)

    def test_result_record(self, action: int, info: dict):
        """
        记录测试结果
        
        参数:
            action: 执行的动作
            info: 包含 true_label 的字典
        """
        pred = 1 if action > 0.5 else 0
        accuracy = 1.0 if pred == int(info['true_label']) else 0.0
        self.testing_accuracy.append(accuracy)