import gymnasium as gym
from gymnasium.wrappers import TransformObservation, NormalizeReward, GrayScaleObservation, ResizeObservation, FlattenObservation
from matplotlib import pyplot as plt
import numpy as np
import random

from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

np.random.seed(0)

env = gym.make("ALE/Breakout-v5", render_mode="human")
env = FlattenObservation(env)
#env = NormalizeReward(env, gamma=0.99, epsilon=1e-8)
#env = GrayScaleObservation(env, keep_dim=True)
#env = ResizeObservation(env, 64)
#env = gym.wrappers.RecordVideo(env, 'test')
env = TransformObservation(env, lambda obs: obs + 0.1 * np.random.randn(*obs.shape))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, n_obs, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_obs, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)
        
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


batch_size=128
gamma=0.99
eps_start=0.9
eps_end=0.05
eps_decay=1000
tau = 0.005
LR = 1e-4

n_actions = env.action_space.n
state, info = env.reset()
#print(len(state)) #100800
n_obs = len(state)

policy_net = DQN(n_obs, n_actions).to(device)
target_net = DQN(n_obs, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.Adam(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(1000000)

stepa_done = 0


def select_action(state):
    global stepa_done
    sample = random.random()
    eps_threshold = eps_end + (eps_start - eps_end) * np.exp(-1. * stepa_done / eps_decay)
    stepa_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)
    

episode_durations = []


def plot_durations(show_results=False):
    plt.figure(1)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated


def optimize():
    """
    Optimizes the policy network using the memory buffer.
    
    This function samples a batch of transitions from the memory buffer and
    computes the loss for the policy network. It then updates the policy 
    network's parameters using the optimizer.
    """
    
    # Check if the memory buffer has enough transitions to form a batch
    if len(memory) < batch_size:
        return
    
    # Sample a batch of transitions from the memory buffer
    transitions = memory.sample(batch_size)
    
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))
    
    # Create a mask for non-final states
    non_final_mask = torch.tensor(tuple(map(lambda  s: s is not None, batch.next_state)), 
                                  device=device, dtype=torch.bool)
    
    # Extract the next states for non-final states
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    
    # Extract the states, actions, and rewards from the batch
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    
    # Compute the Q-values for the current states and actions
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    
    # Compute the maximum Q-values for the next states
    next_state_values = torch.zeros(batch_size, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    
    # Compute the expected Q-values for the current states and actions
    expected_state_action_values = (next_state_values * gamma) + reward_batch
    
    # Compute the loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    
    # Zero out the gradients and perform backpropagation
    optimizer.zero_grad()
    loss.backward()
    
    # Clip the gradients to prevent exploding gradients
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    
    # Update the policy network's parameters
    optimizer.step()
    
    
if torch.cuda.is_available():
    num_episodes = 500
else:
    num_episodes = 50

for i_episode in range(num_episodes):
    # Initialize the environment and state
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    for t in count():
        action = select_action(state)
        observation, reward, terminated, truncated, info = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated
        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
        # Store the transition in memory
        memory.push(state, action, reward, next_state, done)
        state = next_state
        
        optimize()
        
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * tau + target_net_state_dict[key] * (1 - tau)
        target_net.load_state_dict(target_net_state_dict)
        
        if done:
            episode_durations.append(t + 1)
            plot_durations()
            break
        
print("Done")
plot_durations(show_results=True)
plt.ioff()
plt.show()