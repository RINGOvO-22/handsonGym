import math
from typing import Optional
import numpy as np
import gymnasium as gym
# from gymnasium.wrappers import FlattenObservation
import pandas as pd


class creditScoring_v1(gym.Env):

    def __init__(self, mode='train' or 'test'):
        self.mode = mode
        # obsevation space: 10-dimensional vector
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(10,),
            dtype=np.float32
        )

        # action space: discrete action space with 2 actions (0 or 1)
        gym.spaces.Discrete(2)

        # # action space: 1-dimensional vector
        # self.action_space = gym.spaces.Box(
        #     low=0.0,
        #     high=1.0,
        #     shape=(1,),
        #     dtype=np.float32
        # )

        # Define the pointer to the sample for online learning
        self.samplePointer = 0
        # Load the training and test data
        self.train_data, self.test_x, self.test_y = self.load_data()
        # Initialize the empty feature list and target list
        self.train_x = np.empty(shape=(0, self.train_data.shape[1] - 1))
        self.train_y = np.empty(shape=(0,))

        # parameter of the real cost function
        self.cost_weight = np.full(shape=10, fill_value=3)

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

    def strategic_response(self, real_feature, policy_weight: np.ndarray):
        """
        A strategic response function that simulates the applicat's stratigic responce to the model.
        It takes the real feature as input and returns a manipulated feature.

        policy function: WX+b
        """
        return real_feature
    
    # called in .reset() & .step()
    def _get_obs(self):
        if self.mode == 'train':
            observation = self.train_x[self.samplePointer]
        else:
            observation = self.test_x[self.samplePointer]
        return self.strategic_response(observation)
    
    def _get_info(self):
        return

    def reset(self, seed: Optional[int] = 10, options: Optional[dict] = None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

       # shuffle the training data
        shuffled_train = self.train_data.sample(frac=1.0, random_state=42)
        self.train_y = shuffled_train[:, 0].astype(np.float32)
        self.train_x = shuffled_train[:, 1:]

        # Reset the sample pointer to the beginning of the training data
        self.samplePointer = 0

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        self.samplePointer += 1  # Increment the sample pointer
        # The env terminateds when the pointer reaches the end of the data
        if self.mode == 'train':
            terminated = self.samplePointer > len(self.train_x)
        else: 
            terminated = self.samplePointer > len(self.test_x)
        truncated = False
        reward = math.abs(action-self.train_y[self.samplePointer])
        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

# Register the environment after the class definition
gym.register(
    id="creditScoring_v1",
    entry_point="env.creditScoring:creditScoring_v1"
)

env = gym.make("creditScoring_v1")
# env = gym.make('GridWorld-v0', size=5)
# print(f"Original observation space: {env.observation_space}")
# # Flatten the observation space
# wrapped_env = FlattenObservation(env)
# print(f"Flattened observation space: {wrapped_env.observation_space}")
# # Reset the environment
# observation, info = wrapped_env.reset()
# print(f"Initial observation: {observation}")


