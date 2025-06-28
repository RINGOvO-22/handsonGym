import gymnasium as gym
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

# a principal for the credit scoring v1 environment
class Principal_v1:
    def __init__(
        self,
        env: gym.Env,
        learning_rate_critic: float = 0.001,
        learning_rate_actor: float = 0.001,
        learning_rate_cost: float = 0.001,
        init_cost_pram:  float = 0.1
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
        self.policy_weight = np.ones(10+1, dtype=np.float32)  # policy weight for the classifier (w*x + b)
        self.lr_c = learning_rate_critic
        self.q_weights = np.ones(10+1+1, dtype=np.float32) # q value weights for the classifier (v*(s, a) + b)

        # initial cost parameter estimation for the principal
        self.cost_pram_estimation = np.full(shape=10, fill_value=init_cost_pram, dtype=np.float32)
        self.lr_cost = learning_rate_cost

        self.training_error = []

    # policy function
    def get_action(self, obs: np.ndarray) -> int:
        """
        making predictions based on the current observation and take strategic response into account.
        """
        # with probability epsilon return a random action to explore the environment
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        # with probability (1 - epsilon) act greedily (exploit)
        else:
            return int(np.argmax(self.q_values[obs]))

    def update(
        self,
        obs: tuple[int, int, bool],
        action: int,
        reward: float,
        terminated: bool,
        next_obs: tuple[int, int, bool],
    ):
        # """Updates the Q-value of an action."""
        # future_q_value = (not terminated) * np.max(self.q_values[next_obs])
        # temporal_difference = (
        #     reward + self.discount_factor * future_q_value - self.q_values[obs][action]
        # )

        # self.q_values[obs][action] = (
        #     self.q_values[obs][action] + self.lr * temporal_difference
        # )
        # self.training_error.append(temporal_difference)
        """ Update the policy, the q value and the cost parameter estimation """
