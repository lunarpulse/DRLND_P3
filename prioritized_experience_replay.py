import numpy as np
import random
from collections import namedtuple, deque

import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class PrioritizedReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "priority", "done"])
        self.seed = random.seed(seed)
        self.priorities = deque(maxlen=buffer_size)
        self.max_priority = 1.0
    
        self.b = 1.0
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        
        # Init priority for new comming memory
        e = self.experience(state, action, reward, next_state, self.max_priority, done)
        self.memory.append(e)
        self.priorities.append(self.max_priority)
        
    def sample(self, a=1, b=1):
        
        """Sample a batch of experiences from memory by priority."""
        priorities = np.array(self.priorities)
        probs = priorities ** a /  sum(priorities ** a)
        sample_idxes = np.random.choice(np.arange(len(self.memory)), size=self.batch_size, p=probs)
        experiences = []
        for i in sample_idxes:
            experiences.append(self.memory[i])
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        
        is_weights = np.power(len(self.memory) * probs, -self.b)
        is_weights = torch.from_numpy(is_weights).float().to(device)

        return (sample_idxes, states, actions, rewards, next_states, is_weights, dones)
    
    def update(self, memory_idxes, priorities):
        for i, p in zip(memory_idxes, priorities):
            
            state, action, reward, next_state, priority, done = self.memory[i]
            if p > self.max_priority:
                self.max_priority = p
            self.memory[i] = self.experience(state, action, reward, next_state, p, done)
            self.priorities[i] = p
        

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

class WeightedLoss(nn.Module):
    def __init__(self):
        super(WeightedLoss, self).__init__()
        return

    def forward(self, weights, inputs, targets):
        t = torch.abs(inputs - targets)
        zi = torch.where(t < 1, 0.5 * t ** 2, t - 0.5)
        loss = (weights * zi).sum()
        return loss
