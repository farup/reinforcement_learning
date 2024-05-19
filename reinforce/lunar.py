import gymnasium as gym
import gymnasium.wrappers.record_video as RecordVideo
from torch.distributions import Categorical
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from collections import deque
import torch
torch.manual_seed(0)  # set random seed


env = gym.make("LunarLander-v2", render_mode="human")
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

observation, info = env.reset(seed=42)

print(f'Observation space: {state_size}, action space: {action_size}')
print(f'Observation {observation}')


def test_random():
    test_env = gym.make("LunarLander-v2", render_mode="human")

    observation = env.reset()
    for _ in range(2000):
        # Take a random action
        action = test_env.action_space.sample()
        print("Action taken:", action)

        # Do this action in the environment and get
        # next_state, reward, terminated, truncated and info
        observation, reward, terminated, truncated, info = test_env.step(
            action)
        test_env.render()

        # If the game is terminated (in our case we land, crashed) or truncated (timeout)
        if terminated or truncated:
            # Reset the environment
            print("Environment is reset")
            observation, info = test_env.reset()

    test_env.close()


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


class Policy(nn.Module):
    def __init__(self, state_size, action_size, hidden_size):
        super().__init__()
        self.f1 = nn.Linear(state_size, hidden_size)
        self.f2 = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = F.relu(self.f1(x))
        x = F.softmax(self.f2(x), dim=0)
        return x

    def select_action(self, state):
        state = torch.from_numpy(state).float().to(device)
        probs = self.forward(state)
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)


def reinforce(policy, optimizer, n_training_episodes, max_t, gamma, print_every):

    scores = []
    for episode in range(0, n_training_episodes):
        rewards = []
        log_probs = []

        observation, _ = env.reset()
        for i in range(max_t):
            action, log_prob = policy.select_action(observation)

            observation, reward, terminated, truncated, info = env.step(action)
            rewards.append(reward)
            log_probs.append(log_prob)
            if terminated or truncated:
                break

        scores.append(sum(rewards))
        discounted_rewards = deque(maxlen=max_t)
        discounted_reward = 0

        for i in range(len(rewards))[::-1]:
            discounted_reward = gamma*discounted_reward + rewards[i]
            discounted_rewards.appendleft(discounted_reward)

        objective = []
        for log_prob, discounted_reward in zip(log_probs, discounted_rewards):
            objective.append(-log_prob * discounted_reward)

        objective = torch.tensor(objective, requires_grad=True).sum()
        objective.backward()
        optimizer.step()
        del rewards
        del log_probs

        if episode % print_every == 0:
            print(f'Episode {episode}\t Score: {scores[episode]}')

    return scores


lr = 1e-6
gamma = 1.0
n_training_episodes = 100
max_t = 1000

policy = Policy(state_size, action_size, hidden_size=16).to(device)
optimizer = optim.Adam(policy.parameters(), lr)

scores = reinforce(policy, optimizer, n_training_episodes, max_t, gamma, 10)
