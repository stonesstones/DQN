import numpy as np
from dm_control import suite
import torch
import torchvision
import torchvision.transforms as T

def make_env(config):
    if config.domain == "normal":
        return Env(config)
    elif config.domain == "autostop":
        return AutoStopEnv(config)
    elif config.domain == "sparse":
        return SparseEnv(config)
    elif config.domain == "mnist":
        return MnistEnv(config)
    else:
        raise NotImplementedError

class Env:
    def __init__(self, config) -> None:
        self._config = config
        self._obs_space = config.N
        self._act_space = config.N - 1
        self._action = np.array([self._digitized_action(index) for index in np.arange(0, 2**self._act_space)])

    @property
    def obs_spec(self):
        return int(self._obs_space)
    @property
    def act_spec(self):
        return 2**self._act_space
    # 2**self._act_space
    
    def reset(self):
        self.state = np.random.randint(0, 10, size=self._obs_space)
        self.state[0] = np.random.randint(1, 10)
        self.first_state_int = self._arr2int(self.state)
        self.state_int = self.first_state_int
        self.target = np.sort(self.state)
        self.target_int = self._arr2int(self.target)
        return self.state, None, None, self.state_int

    def step(self, action):
        action = self._action[action]
        for index, a in enumerate(action):
            if int(a) == 1:
                tmp_i = self.state[index]
                self.state[index] = self.state[index+1]
                self.state[index+1] = tmp_i
        self.state_int = self._arr2int(self.state)
        reward, done = self._get_reward(self.state_int)
        return self.state, reward, done, self.state_int

    def sample_action(self):
        return np.random.randint(0, 2**self._act_space)

    def _digitized_action(self, action_index):
        index_str = bin(action_index)[2:]
        return np.array([int(i) for i in index_str.zfill(self._act_space)])

    def _get_reward(self, state_int):
        diff = np.abs(self.target_int - state_int)
        if diff == 0:
            return 1, True
        else:
            return -2/(self._obs_space*(self._obs_space - 1)), False

    def _arr2int(self, arr):
        return int("".join([str(i) for i in arr]))

class AutoStopEnv(Env):
    def __init__(self, config) -> None:
        super().__init__(config)

    @property
    def act_spec(self):
        return 2**self._act_space + 1

    def reset(self):
        self.time = 0
        self.state = np.random.randint(0, 10, size=self._obs_space)
        self.state[0] = np.random.randint(1, 10)
        self.first_state_int = self._arr2int(self.state)
        self.state_int = self.first_state_int
        self.target = np.sort(self.state)
        self.target_int = self._arr2int(self.target)
        return self.state, None, None, self.state_int

    def step(self, action):
        terminate = (action == torch.tensor(2**self._act_space))
        if terminate:
            self.state_int = self._arr2int(self.state)
            reward, done = self._get_reward(self.state_int)
        else:
            action = self._action[action]
            for index, a in enumerate(action):
                if int(a) == 1:
                    tmp_i = self.state[index]
                    self.state[index] = self.state[index+1]
                    self.state[index+1] = tmp_i
            self.state_int = self._arr2int(self.state)
            reward, done = 0, False
        return self.state, reward, done, self.state_int
    def sample_action(self):
        return np.random.randint(0, 2**self._act_space+1)

    def _get_reward(self, state_int):
        diff = np.abs(self.target_int - state_int)
        if diff == 0:
            return 1, True
        else:
            return 0, True

class SparseEnv(Env):
    def __init__(self, config) -> None:
        super().__init__(config)
        # self._config = config
        # self._obs_space = config.N
        # self._act_space = config.N - 1
        # self._action = np.array([self._digitized_action(index) for index in np.arange(0, 2**(self._act_space-1))])
        self._time_limit = config.time_limit
    
    def reset(self):
        self.time = 0
        self.state = np.random.randint(0, 10, size=self._obs_space)
        self.state[0] = np.random.randint(1, 10)
        self.first_state_int = self._arr2int(self.state)
        self.state_int = self.first_state_int
        self.target = np.sort(self.state)
        self.target_int = self._arr2int(self.target)
        return self.state, None, None, self.state_int

    def step(self, action):
        action = self._action[action]
        for index, a in enumerate(action):
            if int(a) == 1:
                tmp_i = self.state[index]
                self.state[index] = self.state[index+1]
                self.state[index+1] = tmp_i
        self.state_int = self._arr2int(self.state)
        reward, done = self._get_reward(self.state_int)
        if self.time >= self._time_limit:
            done = True
            self.time = 0
        else:
            self.time += 1
        return self.state, reward, done, self.state_int

    # def sample_action(self):
    #     return np.random.randint(0, 2**self._act_space)

    # def _digitized_action(self, action_index):
    #     index_str = bin(action_index)[2:]
    #     return np.array([int(i) for i in index_str.zfill(self._act_space)])

    def _get_reward(self, state_int):
        diff = np.abs(self.target_int - state_int)
        if diff == 0:
            return 1, True
        else:
            return 0, False

    # def _arr2int(self, arr):
    #     return int("".join([str(i) for i in arr]))

class MnistEnv(Env):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.data = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=T.ToTensor())
        # self.dataloader = torch.utils.data.DataLoader(self.data, batch_size=64)
        self._config = config
        self._obs_space = config.N
        self._act_space = config.N - 1
        self._action = np.array([self._digitized_action(index) for index in np.arange(0, 2**self._act_space)])

    @property
    def obs_spec(self):
        return int(self._obs_space)
    @property
    def act_spec(self):
        return 1
    
    def reset(self):
        self.state = np.random.randint(0, 10, size=self._obs_space)
        self.state[0] = np.random.randint(1, 10)
        self.first_state_int = self._arr2int(self.state)
        self.state_int = self.first_state_int
        self.target = np.sort(self.state)
        self.target_int = self._arr2int(self.target)
        return self.state, None, None, self.state_int

    def step(self, action):
        action = self._action[action]
        for index, a in enumerate(action):
            if int(a) == 1:
                tmp_i = self.state[index]
                self.state[index] = self.state[index+1]
                self.state[index+1] = tmp_i
        self.state_int = self._arr2int(self.state)
        reward, done = self._get_reward(self.state_int)
        return self.state, reward, done, self.state_int

    def sample_action(self):
        return np.random.randint(0, 2**self._act_space)

    def _digitized_action(self, action_index):
        index_str = bin(action_index)[2:]
        return np.array([int(i) for i in index_str.zfill(self._act_space)])

    def _get_reward(self, state_int):
        diff = np.abs(self.target_int - state_int)
        if diff == 0:
            return 1, True
        else:
            return -2/(self._obs_space*(self._obs_space - 1)), False

    def _arr2int(self, arr):
        return int("".join([str(i) for i in arr]))

def main():
    import dataclasses
    @dataclasses.dataclass
    class EnvConfig:
        domain: str = "mnist"
        N: int = 6
        time_limit: int = 100

if __name__ == "__main__":
    main()