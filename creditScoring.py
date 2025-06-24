from typing import Optional
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation


class creditScoring_v1(gym.Env):

    def __init__(self, size: int = 5):
        # The size of the square grid
        self.size = size

        # obsevation space: 10-dimensional vector
        self.observation_space = gym.spaces.Box(
            low=self._get_obs_low(),
            high=self._get_obs_high(),
            shape=(10,),
            dtype=np.float32
        )

        # action space: 2-dimensional vector
        self.action_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32
        )

        # Initializa the environment
        self.state = None


    def strategic_response(self, real_feature):
        """
        A strategic response function that simulates the applicat's stratigic responce to the model.
        It takes the real feature as input and returns a manipulated feature.
        """
        return real_feature
    
    # 在 .reset() 和 .step() 方法中调用, 返回observation
    def _get_obs(self):
        return {"agent": self._agent_location, "target": self._target_location}
    
    #返回 agent 和 target 的曼哈顿距离
    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the agent's location uniformly at random
        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        # We will sample the target's location randomly until it does not coincide with the agent's location
        self._target_location = self._agent_location
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.integers(
                0, self.size, size=2, dtype=int
            )

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[action]
        # We use `np.clip` to make sure we don't leave the grid bounds
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )

        # An environment is completed if and only if the agent has reached the target
        terminated = np.array_equal(self._agent_location, self._target_location)
        truncated = False
        reward = 1 if terminated else 0  # the agent is only reached at the end of the episode
        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

# Register the environment after the class definition
gym.register(
    id="GridWorld-v0",
    entry_point="handsonGym_customEnv:GridWorldEnv",
)

env = gym.make('GridWorld-v0', size=5)
print(f"Original observation space: {env.observation_space}")
# Flatten the observation space
wrapped_env = FlattenObservation(env)
print(f"Flattened observation space: {wrapped_env.observation_space}")
# Reset the environment
observation, info = wrapped_env.reset()
print(f"Initial observation: {observation}")

# analysis: e.g., {'agent': [4, 1], 'target': [2, 4]} → [4, 1, 2, 4]