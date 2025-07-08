import numpy as np

class Principal_v2_PPO:
    def __init__(self, obs_dim=10,
                 lr_actor=1e-3,
                 lr_critic=1e-3,
                 gamma=0.99,
                 lam=0.95,
                 clip_eps=0.1, # 一般 0.1 - 0.2, 低 -> 变化幅度大
                 pos_buffer_size=10,
                 pos_neg_ratio = 100):

        self.obs_dim = obs_dim
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps

        self.lr_actor = lr_actor
        self.lr_critic = lr_critic

        self.actor_weights = np.zeros(obs_dim + 1)
        self.critic_weights = np.zeros(obs_dim + 1)

        self.pos_buffer = []
        self.neg_buffer = []
        self.pos_buffer_size = pos_buffer_size
        self.neg_buffer_size = int(pos_buffer_size * pos_neg_ratio)

        self.training_rewards = []
        self.training_accuracy = []
        self.training_policy_weights = []
        self.training_acc_detail = []
        self.testing_accuracy = []
        self.testing_acc_detail = []

    def get_action(self, obs):
        obs_ext = np.append(obs, 1.0)
        logit = np.dot(self.actor_weights, obs_ext)
        prob = 1 / (1 + np.exp(-logit))
        action = int(np.random.rand() < prob)
        log_prob = action * np.log(prob + 1e-8) + (1 - action) * np.log(1 - prob + 1e-8)
        value = np.dot(self.critic_weights, obs_ext)
        return action, log_prob, value, prob

    def store_transition(self, obs, action, reward, done, next_obs, log_prob, value, true_label):
        sample = (obs, action, reward, done, next_obs, log_prob, value)
        if true_label == 1:
            if len(self.pos_buffer) < self.pos_buffer_size:
                self.pos_buffer.append(sample)
            else:
                idx = np.random.randint(self.pos_buffer_size)
                self.pos_buffer[idx] = sample
        else:
            if len(self.neg_buffer) < self.neg_buffer_size:
                self.neg_buffer.append(sample)
            else:
                idx = np.random.randint(self.neg_buffer_size)
                self.neg_buffer[idx] = sample

        # --- 训练统计记录 ---
        self.training_rewards.append(reward)
        obs_ext = np.append(obs, 1.0)
        prob = 1 / (1 + np.exp(-np.dot(self.actor_weights, obs_ext)))
        pred = int(prob > 0.5)
        acc = 1.0 if pred == int(true_label) else 0.0
        self.training_accuracy.append(acc)
        self.training_policy_weights.append(self.actor_weights.copy())
        self.training_acc_detail.append({
            'predicted_prob': prob,
            'predicted_label': pred,
            'true_label': true_label,
            'accuracy': acc
        })

        if len(self.pos_buffer) >= self.pos_buffer_size and len(self.neg_buffer) >= self.neg_buffer_size:
            self.batch_update()

    def batch_update(self):
        batch = self.pos_buffer + self.neg_buffer
        np.random.shuffle(batch)

        obs_list, action_list, reward_list, done_list, next_obs_list, logp_old_list, value_list = zip(*batch)

        obs_mat = np.array([np.append(o, 1.0) for o in obs_list])
        next_obs_mat = np.array([np.append(o, 1.0) for o in next_obs_list])
        actions = np.array(action_list)
        rewards = np.array(reward_list)
        dones = np.array(done_list).astype(float)
        old_log_probs = np.array(logp_old_list)
        values = np.array(value_list)

        next_values = np.dot(next_obs_mat, self.critic_weights)
        td_targets = rewards + self.gamma * next_values * (1 - dones)
        advantages = td_targets - values

        # policy update (PPO style)
        logits = np.dot(obs_mat, self.actor_weights)
        probs = 1 / (1 + np.exp(-logits))
        new_log_probs = actions * np.log(probs + 1e-8) + (1 - actions) * np.log(1 - probs + 1e-8)

        ratios = np.exp(new_log_probs - old_log_probs)
        clipped_ratios = np.clip(ratios, 1 - self.clip_eps, 1 + self.clip_eps)
        loss_grad = -np.mean((np.minimum(ratios, clipped_ratios) * advantages).reshape(-1, 1) * obs_mat, axis=0)
        self.actor_weights -= self.lr_actor * loss_grad

        # value update (TD target)
        td_errors = td_targets - np.dot(obs_mat, self.critic_weights)
        critic_grad = -np.mean(td_errors.reshape(-1, 1) * obs_mat, axis=0)
        self.critic_weights -= self.lr_critic * critic_grad

        self.pos_buffer.clear()
        self.neg_buffer.clear()

    def test_result_record(self, action, info, prob):
        accuracy = 1.0 if action == int(info["true_label"]) else 0.0
        self.testing_accuracy.append(accuracy)
        self.testing_acc_detail.append({
            'action': action,
            'true_label': info['true_label'],
            'prob': prob
        })
