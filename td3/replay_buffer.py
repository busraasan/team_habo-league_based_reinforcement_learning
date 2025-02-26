import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import namedtuple, deque
import pickle

# --------------------------------------------------------------------------
# 1. Data Structures: Segment Trees & Prioritized Replay
# --------------------------------------------------------------------------
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

# If you have your own "SumSegmentTree" and "MinSegmentTree" in utils.py, import them:
# from utils import SumSegmentTree, MinSegmentTree

class SumSegmentTree:
    """Simple sum segment tree for prioritized replay (must have capacity = power of 2)."""
    def __init__(self, capacity):
        self._capacity = capacity
        self._tree = [0.0] * (2 * capacity)
    
    def __setitem__(self, idx, val):
        # idx in [0, capacity)
        idx += self._capacity
        self._tree[idx] = val
        idx //= 2
        while idx >= 1:
            self._tree[idx] = self._tree[2*idx] + self._tree[2*idx+1]
            idx //= 2

    def __getitem__(self, idx):
        return self._tree[idx + self._capacity]

    def sum(self, start, end):
        """Returns sum of [start, end]."""
        if end < start: return 0.0
        start += self._capacity
        end += self._capacity
        s = 0.0
        while start <= end:
            if (start % 2) == 1:
                s += self._tree[start]
                start += 1
            if (end % 2) == 0:
                s += self._tree[end]
                end -= 1
            start //= 2
            end //= 2
        return s

    def find_prefixsum_idx(self, prefixsum):
        """Find highest i in [0, size-1] with cumsum(tree[i]) <= prefixsum."""
        idx = 1
        while idx < self._capacity:
            left = 2 * idx
            if self._tree[left] > prefixsum:
                idx = left
            else:
                prefixsum -= self._tree[left]
                idx = left + 1
        return idx - self._capacity

    @property
    def capacity(self):
        return self._capacity


class MinSegmentTree:
    """Simple min segment tree for prioritized replay (capacity = power of 2)."""
    def __init__(self, capacity):
        self._capacity = capacity
        self._tree = [float('inf')] * (2 * capacity)
    
    def __setitem__(self, idx, val):
        idx += self._capacity
        self._tree[idx] = val
        idx //= 2
        while idx >= 1:
            self._tree[idx] = min(self._tree[2*idx], self._tree[2*idx+1])
            idx //= 2

    def __getitem__(self, idx):
        return self._tree[idx + self._capacity]

    def min(self, start, end):
        """Returns min of [start, end]."""
        if end < start: return float('inf')
        start += self._capacity
        end += self._capacity
        m = float('inf')
        while start <= end:
            if (start % 2) == 1:
                m = min(m, self._tree[start])
                start += 1
            if (end % 2) == 0:
                m = min(m, self._tree[end])
                end -= 1
            start //= 2
            end //= 2
        return m

    @property
    def capacity(self):
        return self._capacity

class PrioritizedReplayBuffer:
    """
    Prioritized Replay Buffer using Segment Trees for O(log N) updates & sampling.
    """
    def __init__(self, capacity, alpha=0.6):
        """
        Args:
            capacity (int): Must be a power-of-2 for these segment trees.
            alpha (float): How much prioritization is used (0 = no prioritization, 1 = full).
        """
        self.capacity = capacity
        self.alpha = alpha
        self.pos = 0
        self.size = 0
        self.max_priority = 1.0  # Keep track of max priority so new samples have that priority
        self.transitions = [None] * capacity

        self.sum_tree = SumSegmentTree(capacity)
        self.min_tree = MinSegmentTree(capacity)

    def add_transition(self, transition: Transition):
        idx = self.pos
        self.transitions[idx] = transition
        # Initialize with max priority to ensure it is sampled at least once
        self.sum_tree[idx] = self.max_priority ** self.alpha
        self.min_tree[idx] = self.max_priority ** self.alpha

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, beta=0.4):
        """
        Sample a batch of transitions, with probabilities ~ priority^alpha.
        Returns: (list_of_Transition, weights, indices)
        """
        batch_size = min(batch_size, self.size)
        indices = []
        total_p = self.sum_tree.sum(0, self.size - 1)
        segment = total_p / batch_size

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            idx = self.sum_tree.find_prefixsum_idx(s)
            indices.append(idx)

        transitions = [self.transitions[idx] for idx in indices]

        # Compute IS weights
        sum_of_priorities = self.sum_tree.sum(0, self.size - 1)
        min_p = self.min_tree.min(0, self.size - 1) / sum_of_priorities
        max_weight = (min_p * self.size) ** (-beta)

        weights = []
        for idx in indices:
            p_i = self.sum_tree[idx] / sum_of_priorities
            w_i = (p_i * self.size) ** (-beta)
            weights.append(w_i / max_weight)
        weights = np.array(weights, dtype=np.float32)

        return transitions, weights, indices

    def update_priorities(self, indices, priorities):
        """Update the priorities of sampled transitions."""
        for idx, priority in zip(indices, priorities):
            priority = max(priority, 1e-6)
            self.sum_tree[idx] = priority ** self.alpha
            self.min_tree[idx] = priority ** self.alpha
            if priority > self.max_priority:
                self.max_priority = priority

    def __len__(self):
        return self.size
    
        

class ReplayMemory():
    '''
        Basic replay memory for DQN agents.
    '''
    def __init__(self, capacity=100000):
        self.memory = deque([], maxlen=capacity)

    def add_transition(self, transition: Transition):
        self.memory.append(transition)

    def sample(self, batch_size):
        if batch_size > self.__len__():
            batch_size = self.__len__()
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)