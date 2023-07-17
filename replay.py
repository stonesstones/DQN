import torch
import numpy as np
import random


class ReplayBuffer:

    def __init__(self, capacity, obs_shape, action_size, batch_size):
        self.capacity = capacity
        self.obs_shape = obs_shape
        self.action_size = action_size
        self.batch_size = batch_size
        self.idx = 0
        self.full = False
        self.state = np.empty((capacity, obs_shape), dtype=np.float32)
        self.next_state = np.empty((capacity, obs_shape), dtype=np.float32)
        self.actions = np.empty((capacity, action_size), dtype=np.float32)
        self.rewards = np.empty((capacity,), dtype=np.float32) 
        self.done = np.empty((capacity,), dtype=bool)
        self.steps, self.episodes = 0, 0
    
    def add(self, global_step, state, ac, rew, next_state, done):
        self.state[self.idx] = state
        self.actions[self.idx] = ac
        self.rewards[self.idx] = rew
        self.next_state[self.idx] = next_state
        self.done[self.idx] = done
        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0
        self.steps += 1 
        self.episodes = self.episodes + (1 if next_state == None else 0)

    def sample(self):
        sample_idxs = np.random.randint(0, self.capacity if self.full else self.idx, self.batch_size)
        state,acs,rews,next_state,done = self.state[sample_idxs], \
                                        self.actions[sample_idxs],\
                                        self.rewards[sample_idxs],\
                                        self.next_state[sample_idxs],\
                                        self.done[sample_idxs],
        return state,acs,rews,next_state,done