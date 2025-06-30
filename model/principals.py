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
        init_cost_pram:  float = 2 # same as in the "made practical" paper 
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
        self.lr_a = learning_rate_actor
        self.previous_policy_weight = np.ones(10+1, dtype=np.float32)  # policy weight for the classifier sigmoid(w*x + b)
        self.lr_c = learning_rate_critic
        self.q_weights = np.ones(10+1+1, dtype=np.float32) # q value weights for the classifier (v*(s, a) + b)

        # initial cost parameter estimation for the principal
        self.cost_pram_estimation = np.full(shape=10, fill_value=init_cost_pram, dtype=np.float32)
        self.lr_cost = learning_rate_cost

        self.training_error = []
        self.training_accuracy = []

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
        """ Update the policy, the q value function parameter and the cost parameter estimation """

        """
        使用真实标签 info 更新当前策略模型（sigmoid(w^T x + b)）
        
        参数:
            obs (np.ndarray): 当前观察值（特征向量）
            action (int): 执行的动作（暂时未使用）
            reward (float): 奖励（暂时未使用）
            terminated (bool): 是否终止
            info (dict): 包含真实标签的字典，通常是 {'true_label': int}，其中 int 是 0 或 1
        """
        # Step 1: 添加 bias 到 obs
        obs_with_bias = np.append(obs, 1.0)  # shape: (11,)
        
        # Step 2: 计算 logits 和 sigmoid 概率
        logits = np.dot(self.previous_policy_weight, obs_with_bias)
        prob = 1 / (1 + np.exp(-logits))  # sigmoid
        
        # Step 3: 计算梯度（Binary Cross Entropy 的梯度）
        gradient = (prob - info['true_label']) * obs_with_bias
        
        # Step 4: 更新 policy weight
        self.previous_policy_weight -= self.lr_a * gradient
        
        # Step 5: 可选：记录训练误差
        self.training_error.append(abs(prob - info['true_label']))

        pred = 1 if prob > 0.5 else 0
        accuracy = 1.0 if pred == int(info['true_label']) else 0.0
        self.training_accuracy.append(accuracy)