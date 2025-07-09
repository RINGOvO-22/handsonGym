import math
from typing import Optional
import numpy as np
import gymnasium as gym
# from gymnasium.wrappers import FlattenObservation
import pandas as pd
import torch
from tqdm import tqdm

# hyperparameters
test_label_threshold = 0.5  # threshold for the test label
seed = 77
strategic_response = False

class creditScoring_v1(gym.Env):

    def __init__(self, mode='train' or 'test', 
                 policy_weight=[0.1]*11, 
                 maximum_episode_length: int = 1000000):
        self.mode = mode
        self.maximum_episode_length = maximum_episode_length

        # obsevation space: 10-dimensional vector
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            # high = 10000000,
            shape=(10,),
            dtype=np.float64
        )

        # action space: discrete action space with 2 actions (0 or 1)
        self.action_space = gym.spaces.Discrete(2)

        # Define the pointer to the sample for online learning
        self.samplePointer = 0
        # Load the training and test data
        self.train_data, self.test_x, self.test_y = self.load_data()
        # Initialize the empty feature list and target list
        self.train_x = np.empty(shape=(0, self.train_data.shape[1] - 1))
        self.train_y = np.empty(shape=(0,))

        # parameter of the real cost function
        # Assume using a weighted quadratic cost function (same as in the "made practical" paper)
        self.cost_weight = np.full(shape=10, fill_value=0.5, dtype=np.float64)
        self.policy_weight = policy_weight

    def load_data(self):
        path = "data/ProcessedData/"

        train_data = pd.read_csv(path + "cs-training-processed.csv")
        # train_data["NumberOfDependents"] = train_data["NumberOfDependents"].astype(int)
        # train_data["MonthlyIncome"] = train_data["MonthlyIncome"].astype(int)

        test_data = pd.read_csv(path + "cs-test-processed.csv")
        test_prob = pd.read_csv(path + "sampleEntry.csv")
        # test_data["NumberOfDependents"] = test_data["NumberOfDependents"].astype(int)
        # test_data["MonthlyIncome"] = test_data["MonthlyIncome"].astype(int)
        # extract the target column and the features
        test_x = test_data.drop(columns=['SeriousDlqin2yrs']).to_numpy()
        test_y = test_prob.drop(columns=['Id']).to_numpy()  # drop the ID column
        test_y = test_y.ravel() # flatten the target to 1D array
        test_y = (test_y >= test_label_threshold).astype(int)
        
        print("test_y label distribution:", np.bincount(test_y))
        
        # test_data_2 = np.column_stack((test_x, test_y))  # combine features and target for test data
        
        # print(f"train_x shape: {train_x.shape}, dtype: {train_x.dtype}")
        # print(f"train_y shape: {train_y.shape}, dtype: {train_y.dtype}")
        # print(f"test_x shape: {test_x.shape}, dtype: {test_x.dtype}")
        # print(f"test_y shape: {test_y.shape}, dtype: {test_y.dtype}")

        return train_data, test_x, test_y
    
    def strategic_response(self, 
                           real_feature: np.ndarray, 
                           policy_weight: np.ndarray,
                           learning_rate=0.01,
                           num_steps=50):
        """
        A strategic response function that simulates the applicant's strategic response to the model.
        """
        # 统计调用次数（静态变量实现）
        if not hasattr(self, "_response_call_count"):
            self._response_call_count = 0
        self._response_call_count += 1

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        real_feature = real_feature.astype(np.float64)
        if not strategic_response:
            return real_feature

        policy_weight = np.array(policy_weight).astype(np.float64)
        cost_weight = self.cost_weight.astype(np.float64)

        real_x = torch.tensor(real_feature, device=device, requires_grad=False)
        cost_v = torch.tensor(cost_weight, device=device, requires_grad=False)
        W = torch.tensor(policy_weight[:-1], device=device, requires_grad=False)
        b = torch.tensor(policy_weight[-1], device=device, requires_grad=False)
        z = torch.tensor(real_feature, device=device, requires_grad=True)

        optimizer = torch.optim.Adam([z], lr=learning_rate)

        # 如果需要记录 z 的变化曲线
        record_z = self._response_call_count in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        if record_z:
            z_history = []

        for _ in range(num_steps):
            optimizer.zero_grad()
            logits = torch.dot(W, z) + b
            fz = torch.sigmoid(logits)
            cz = torch.sum(cost_v * (z - real_x) ** 2)
            loss = -fz + cz
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                z[:] = z.clamp(0.0, 1.0)

            if record_z:
                z_history.append(z.detach().cpu().numpy().copy())

        # 可视化：画出每一维 z 的变化趋势
        if record_z:
            import matplotlib.pyplot as plt
            z_array = np.array(z_history)  # shape: (steps, dim)
            dim = z_array.shape[1]
            plt.figure(figsize=(12, 6))
            for i in range(dim):
                plt.plot(range(num_steps), z_array[:, i], label=f'z[{i}]')
            plt.title(f'z Convergence Trend (Call #{self._response_call_count})')
            plt.xlabel("Iteration Step")
            plt.ylabel("z value")
            plt.legend(loc='best', bbox_to_anchor=(1.05, 1))
            plt.tight_layout()
            plt.grid(True)
            plt.savefig(f"./result/last_experiment/z_convergence_call_{self._response_call_count}.png")
            # plt.show()
            plt.close()

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

        next_obs = self.strategic_response(self.train_x[self.samplePointer+1], self.policy_weight)
        return {'true_label': target, 'next_obs': next_obs}

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # shuffle the training data
        shuffled_train = self.train_data.sample(frac=1.0, random_state=seed)

        # 使用 iloc 按位置提取第一列作为标签
        self.train_y = shuffled_train.iloc[:, 0].astype(np.float64).values

        # 剩余列作为特征
        self.train_x = shuffled_train.iloc[:, 1:].to_numpy()

        # Reset the sample pointer to the beginning of the training data
        self.samplePointer = 0

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action, previous_policy_weight=None):
        # test
        # print(f"\nSample pointer: {self.samplePointer}")
        # print(f"total samples: {len(self.train_x) if self.mode == 'train' else len(self.test_x)}")
        # print(f"\nSampling persentage: {math.ceil(self.samplePointer/len(self.train_x)*100)}%")
        
        if action == 0 and self.train_y[self.samplePointer] == 0:
            reward = +1  # 正确批准
        elif action == 0 and self.train_y[self.samplePointer] == 1:
            reward = -1  # 错误批准
        elif action == 1 and self.train_y[self.samplePointer] == 0:
            reward = -1  # 错误拒绝好用户
        else:  # action == 1 and label == 1
            reward = +1  # 正确拒绝坏用户
            
        self.policy_weight = previous_policy_weight if previous_policy_weight is not None else self.policy_weight
        
        # The env terminateds when the pointer reaches the end of the data
        if self.mode == 'train':
            terminated = self.samplePointer > len(self.train_x)
        else: 
            terminated = self.samplePointer > len(self.test_x)
        truncated = self.samplePointer > self.maximum_episode_length
        
        if not (terminated or truncated):
            info = self._get_info()
            self.samplePointer += 1
            next_obs = self._get_obs()
        else:
            next_obs = None
            info = {}

        return next_obs, reward, terminated, truncated, info

# Register the environment after the class definition
gym.register(
    id="creditScoring_v1",
    entry_point="env.creditScoring:creditScoring_v1"
)
