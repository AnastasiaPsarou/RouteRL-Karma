import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim

from collections import deque
from .learning_model import BaseLearningModel

class DQN(BaseLearningModel):
    def __init__(self, state_size,
                action_space_size,
                epsilon=0.99,
                epsilon_decay_rate=0.01,
                epsilon_min=0.00,
                memory_size=1000,
                batch_size=32,
                gamma=0.9,
                learning_rate=0.003,
                widths=(32, 64, 32),
                target_update_freq=100,
                device=None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_size = state_size
        self.action_space_size = action_space_size
        self.epsilon = epsilon
        self.epsilon_decay_rate = epsilon_decay_rate
        self.epsilon_min = epsilon_min
        #self.memory = ReplayBuffer(maxlen=memory_size)
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.widths = widths
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        self.learn_step = 0  # count how many gradient steps we've done

        self.q_network = Network(self.state_size, self.action_space_size, self.widths).to(self.device)
        
        self.target_q_network = Network(self.state_size, self.action_space_size, self.widths).to(self.device)
        self.update_target_network()  # initial hard copy
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()

        self.loss = list()

    def train(self):
        """Set the model to training mode"""
        self.q_network.train()
        return self

    def update_target_network(self):
        """Copy weights from online network to target network (hard update)."""
        self.target_q_network.load_state_dict(self.q_network.state_dict())


    def eval(self):
        """Set the model to evaluation mode"""
        self.q_network.eval()
        return self

    def act(self, state):
        if self.q_network.training and np.random.rand() < self.epsilon:
            action = torch.randint(0, self.action_space_size, (1,)).item()
            return action
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            action = torch.argmax(q_values).item()
            
            self.last_state = state
            self.last_action = action

            return action

    def learn(self, rollouts, batch_size):

        """batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_tensor = torch.FloatTensor(states).to(self.device)
        actions_tensor = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states_tensor = torch.FloatTensor(next_states).to(self.device)
        dones_tensor = torch.FloatTensor(dones).unsqueeze(1).to(self.device)"""
        
        """
        rollouts: tuple of 5 numpy arrays:
            (states, actions, rewards, next_states, dones)
            - states:      shape (N, state_dim)
            - actions:     shape (N,)
            - rewards:     shape (N,)
            - next_states: shape (N, state_dim)
            - dones:       shape (N,)
        batch_size: int
        """

        states, actions, rewards, next_states, dones = rollouts
        N = len(actions)

        # Sample a mini-batch from the rollouts
        # (with replacement if batch_size > N)
        indices = np.random.choice(N, size=batch_size, replace=(batch_size > N))

        batch_states      = states[indices]
        batch_actions     = actions[indices]
        batch_rewards     = rewards[indices]
        batch_next_states = next_states[indices]
        batch_dones       = dones[indices]

        # Convert to tensors
        states_tensor      = torch.as_tensor(batch_states, dtype=torch.float32, device=self.device)
        actions_tensor     = torch.as_tensor(batch_actions, dtype=torch.long,    device=self.device).unsqueeze(1)
        rewards_tensor     = torch.as_tensor(batch_rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states_tensor = torch.as_tensor(batch_next_states, dtype=torch.float32, device=self.device)
        dones_tensor       = torch.as_tensor(batch_dones, dtype=torch.float32,   device=self.device).unsqueeze(1)

        current_q_values = self.q_network(states_tensor).gather(1, actions_tensor)

        with torch.no_grad():
            next_q_values = self.target_q_network(next_states_tensor).max(1, keepdim=True)[0]
            target_q_values = rewards_tensor + self.gamma * (1 - dones_tensor) * next_q_values

        loss = self.loss_fn(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Increment learn step counter and periodically update target network
        self.learn_step += 1
        if self.learn_step % self.target_update_freq == 0:
            self.update_target_network()

        self.decay_epsilon()
        self.loss.append(loss.item())

    def decay_epsilon(self):
        print("epsilon is: ", self.epsilon, "\n\n")
        self.epsilon = max(self.epsilon - self.epsilon_decay_rate, self.epsilon_min)

class Network(nn.Module):
    def __init__(self, state_size, action_space_size, widths):
        super(Network, self).__init__()
        
        self.input_layer = nn.Linear(state_size, widths[0])
        self.hidden_layers = nn.ModuleList([nn.Linear(widths[x], widths[x+1]) for x in range(len(widths) - 1)])
        self.out_layer = nn.Linear(widths[-1], action_space_size)

    def forward(self, x):
        x = torch.relu(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            x = torch.relu(hidden_layer(x))
        x = self.out_layer(x)
        return x


class ReplayBuffer:
    def __init__(self, capacity: int):
        """
        Simple replay buffer for off-policy algorithms like DQN.

        Args:
            capacity (int): Maximum number of transitions to store.
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """
        Store a single transition in the buffer.

        Args:
            state:        state at time t
            action:       action taken at time t
            reward:       reward received after taking action
            next_state:   state at time t+1
            done:         episode done flag (bool or 0/1)
        """
        #print("buffer.append\n\n")
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        """
        Sample a batch of transitions.

        Returns:
            states, actions, rewards, next_states, dones
            Each is a NumPy array of shape (batch_size, ...)
        """
        print("buffer size is: ", len(self.buffer), "\n\n")
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states      = np.array(states, dtype=np.float32)
        actions     = np.array(actions, dtype=np.int64)
        rewards     = np.array(rewards, dtype=np.float32)
        next_states = np.array(next_states, dtype=np.float32)
        dones       = np.array(dones, dtype=np.float32)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        """Return current size of internal memory."""
        return len(self.buffer)