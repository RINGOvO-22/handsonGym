import math
from typing import Optional
import numpy as np
import gymnasium as gym
# from gymnasium.wrappers import FlattenObservation
import pandas as pd
import torch
from tqdm import tqdm


class creditScoring_v1(gym.Env):

    def __init__(self, mode='train' or 'test', policy_weight=[1]*11):
        self.mode = mode
        # obsevation space: 10-dimensional vector
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(10,),
            dtype=np.float32
        )

        # action space: discrete action space with 2 actions (0 or 1)
        # self.action_space = gym.spaces.Discrete(2)

        # action space: 1-dimensional vector
        self.action_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32
        )

        # Define the pointer to the sample for online learning
        self.samplePointer = 0
        # Load the training and test data
        self.train_data, self.test_x, self.test_y = self.load_data()
        # Initialize the empty feature list and target list
        self.train_x = np.empty(shape=(0, self.train_data.shape[1] - 1))
        self.train_y = np.empty(shape=(0,))

        # parameter of the real cost function
        # Assume using a weighted quadratic cost function (same as in the "made practical" paper)
        self.cost_weight = np.full(shape=10, fill_value=0.5, dtype=np.float32)
        self.policy_weight = policy_weight

    def load_data(self):
        path = "data/ProcessedData/"

        train_data = pd.read_csv(path + "cs-training-processed.csv")
        train_data["NumberOfDependents"] = train_data["NumberOfDependents"].astype(int)
        train_data["MonthlyIncome"] = train_data["MonthlyIncome"].astype(int)
        # extract the target column and the features
        # train_x = train_data.drop(columns=['SeriousDlqin2yrs']).to_numpy()
        # train_y = train_data['SeriousDlqin2yrs'].to_numpy()  # extract the target column

        test_data = pd.read_csv(path + "cs-test-processed.csv")
        test_prob = pd.read_csv(path + "sampleEntry.csv")
        test_data["NumberOfDependents"] = test_data["NumberOfDependents"].astype(int)
        test_data["MonthlyIncome"] = test_data["MonthlyIncome"].astype(int)
        # extract the target column and the features
        test_x = test_data.drop(columns=['SeriousDlqin2yrs']).to_numpy()
        test_y = test_prob.drop(columns=['Id']).to_numpy()  # drop the ID column
        test_y = test_y.ravel() # flatten the target to 1D array
        
        # test_data_2 = np.column_stack((test_x, test_y))  # combine features and target for test data
        
        # print(f"train_x shape: {train_x.shape}, dtype: {train_x.dtype}")
        # print(f"train_y shape: {train_y.shape}, dtype: {train_y.dtype}")
        # print(f"test_x shape: {test_x.shape}, dtype: {test_x.dtype}")
        # print(f"test_y shape: {test_y.shape}, dtype: {test_y.dtype}")

        return train_data, test_x, test_y

    # def strategic_response(self, 
    #                        real_feature: np.ndarray, 
    #                        policy_weight: np.ndarray,
    #                        learning_rate=0.01,
    #                        num_steps=50):
    #     """
    #     A strategic response function that simulates the applicat's stratigic responce to the model.
    #     It takes the real feature as input and returns a manipulated feature.
    #     policy function: sigmoid(WX+b)
    #     成本函数: c(z, x) = sum(v_i * (z_i - x_i)^2)
    #     参数:
    #         real_feature (np.ndarray): 原始特征 x (numpy array)
    #         policy_weight (np.ndarray): 当前模型权重 W 和 bias b (最后一项是 bias)
    #         cost_weights (np.ndarray): 成本函数中的 v_i 参数
    #         learning_rate (float): 梯度下降的学习率
    #         num_steps (int): 优化步数
            
    #     返回:
    #         z (np.ndarray): 最优响应后的修改特征 z*
    #     """
    #     # 检查是否可用 GPU
    #     device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #     # print(f"Using device: {device}")

    #     # # test
    #     # device = 'cpu'  # 强制使用 CPU 以避免 GPU 相关问题

    #     # # 转换为 float32 类型
    #     real_feature = real_feature.astype(np.float32)
    #     policy_weight = np.array(policy_weight).astype(np.float32)
    #     cost_weight = self.cost_weight.astype(np.float32)

    #     # 转为 PyTorch tensor 并移动到 GPU（如果可用）
    #     real_x = torch.tensor(real_feature, device=device, requires_grad=False)
    #     cost_v = torch.tensor(cost_weight, device=device, requires_grad=False)
    #     W = torch.tensor(policy_weight[:-1], device=device, requires_grad=False)  # 权重
    #     b = torch.tensor(policy_weight[-1], device=device, requires_grad=False)   # 偏置

    #     # 初始化 z 为原始输入，并设置 requires_grad=True 以进行梯度优化
    #     z = torch.tensor(real_feature, device=device, requires_grad=True)

    #     # 优化器（也可以换成 SGD、AdamW 等）
    #     optimizer = torch.optim.Adam([z], lr=learning_rate)

    #     with tqdm(total=num_steps, desc="Best response: Optimizing z", leave=False) as pbar:
    #         for _ in range(num_steps):
    #             optimizer.zero_grad()

    #             # f(z) = sigmoid(W·z + b)
    #             logits = torch.dot(W, z) + b
    #             fz = torch.sigmoid(logits)

    #             # c(z, x) = sum(v_i * (z_i - x_i)^2)
    #             cz = torch.sum(cost_v * (z - real_x) ** 2)

    #             # 总目标：minimize f(z) + c(z, x)
    #             loss = fz + cz

    #             # 反向传播和优化
    #             loss.backward()
    #             optimizer.step()

    #             # Clip 到 [0, 1] 区间（假设输入归一化过）
    #             with torch.no_grad():
    #                 z[:] = z.clamp(0.0, 1.0)

    #     # 返回 numpy 格式结果
    #     return z.detach().cpu().numpy()
    
    # 带调试信息
    def strategic_response(self, 
                       real_feature: np.ndarray, 
                       policy_weight: np.ndarray,
                       learning_rate=0.01,
                       num_steps=50):
        """
        A strategic response function that simulates the applicant's strategic response to the model.
        """

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        real_feature = real_feature.astype(np.float32)
        policy_weight = np.array(policy_weight).astype(np.float32)
        cost_weight = self.cost_weight.astype(np.float32)

        real_x = torch.tensor(real_feature, device=device, requires_grad=False)
        cost_v = torch.tensor(cost_weight, device=device, requires_grad=False)
        W = torch.tensor(policy_weight[:-1], device=device, requires_grad=False)
        b = torch.tensor(policy_weight[-1], device=device, requires_grad=False)

        z = torch.tensor(real_feature, device=device, requires_grad=True)
        optimizer = torch.optim.Adam([z], lr=learning_rate)

        try:
            with tqdm(total=num_steps, desc="Optimizing z", leave=False) as pbar:
                for step in range(num_steps):
                    optimizer.zero_grad()

                    logits = torch.dot(W, z) + b
                    fz = torch.sigmoid(logits)
                    cz = torch.sum(cost_v * (z - real_x) ** 2)
                    loss = fz + cz

                    loss.backward()
                    optimizer.step()

                    with torch.no_grad():
                        z[:] = z.clamp(0.0, 1.0)

                    pbar.set_postfix(loss=f"{loss.item():.6f}")
                    pbar.update(1)
                    pbar.refresh()

        except Exception as e:
            print(f"[ERROR] 在优化过程中发生异常: {e}")
            import traceback
            traceback.print_exc()
            raise

        return z.detach().cpu().numpy()

    # called in .reset() & .step()
    def _get_obs(self):
        if self.mode == 'train':
            observation = self.train_x[self.samplePointer]
        else:
            observation = self.test_x[self.samplePointer]
        return self.strategic_response(observation, self.policy_weight)
    
    def _get_info(self):
        if self.mode == 'train':
            target = self.train_y[self.samplePointer]
        else:
            target = self.test_y[self.samplePointer]
        return {'true_label': target}

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # shuffle the training data
        shuffled_train = self.train_data.sample(frac=1.0, random_state=42)

        # 使用 iloc 按位置提取第一列作为标签
        self.train_y = shuffled_train.iloc[:, 0].astype(np.float32).values

        # 剩余列作为特征
        self.train_x = shuffled_train.iloc[:, 1:].to_numpy()

        # Reset the sample pointer to the beginning of the training data
        self.samplePointer = 0

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        self.samplePointer += 1  # Increment the sample pointer
        # test
        print(f"\nSample pointer: {self.samplePointer}")
        print(f"total samples: {len(self.train_x) if self.mode == 'train' else len(self.test_x)}")
        print(f"\nSampling persentage: {math.ceil(self.samplePointer/len(self.train_x)*100)}%")
        # The env terminateds when the pointer reaches the end of the data
        if self.mode == 'train':
            terminated = self.samplePointer > len(self.train_x)
        else: 
            terminated = self.samplePointer > len(self.test_x)
        truncated = False
        reward = abs(action-self.train_y[self.samplePointer])
        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

# Register the environment after the class definition
gym.register(
    id="creditScoring_v1",
    entry_point="env.creditScoring:creditScoring_v1"
)