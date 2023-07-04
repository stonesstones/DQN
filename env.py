import numpy as np
from dm_control import suite

class DMEnv:
    def __init__(self, config) -> None:
        self._config = config
        self._env = suite.load(domain_name=config.domain, task_name=config.task)
        self._obs_space = np.linspace(-1, 1, self._config.obs_discrete+1)[1:-1]
        self._act_space = np.linspace(self._env.action_spec().minimum[0], 
                                     self._env.action_spec().maximum[0], 
                                     self._config.act_discrete)

        self._digitized_obs = np.arange(self._config.obs_discrete**2)
        self._digitized_act = np.arange(self._config.act_discrete)
        
    
    @property
    def obs_spec(self):
        return int(self._config.obs_discrete**2)
    @property
    def act_spec(self):
        return int(self._config.act_discrete)
    
    def reset(self):
        obs = self._env.reset()
        self._env.physics.named.data.qpos["hinge"] = 0.
        obs = self._env.step(0)
        reward = self._get_reward(obs.observation)
        digitized_obs_index = self._digitize_obs(obs.observation)
        digitized_obs = np.zeros(self._config.obs_discrete**2)
        digitized_obs[digitized_obs_index] = 1
        return digitized_obs, None, None, None

    def _get_reward(self, obs):
        d = self._config.obs_discrete
        vec, vel = obs['orientation'], obs['velocity']
        rad = np.arctan2(vec[1], vec[0])
        normalized_rad = rad / np.pi # [-1, 1]
        normalized_vel = np.clip(vel, -8, 8)/8
        n_best = (d+1)/2.
        n_rad = np.digitize(normalized_rad, np.linspace(-1, 1, d+1)[1:-1])
        n_vel = np.digitize(normalized_vel, np.linspace(-1, 1, d+1)[1:-1])

        return -(((n_rad-n_best)/d)**2 + 0.01*((n_vel-n_best)/d)**2)
        # reward = -(normalized_rad**2 + 0.01*normalized_vel**2)
        # if np.abs(normalized_rad) < 0.5:
        #     reward += 1
        # elif np.abs(normalized_rad) < 0.2:
        #     reward += 2
        # elif np.abs(normalized_rad) < 0.1:
        #     reward += 3
        # return reward

    def step(self, action):
        real_action = self._act_space[action]
        obs = self._env.step(real_action)
        obs = self._env.step(real_action)
        reward = self._get_reward(obs.observation)
        digitized_obs_index = self._digitize_obs(obs.observation)
        digitized_obs = np.zeros(self._config.obs_discrete**2)
        digitized_obs[digitized_obs_index] = 1
        return digitized_obs, reward, None, None
    
    def _digitize_obs(self, obs):
        vec, vel = obs['orientation'], obs['velocity']
        rad = np.arctan2(vec[1], vec[0])
        normalized_rad = rad / np.pi # [-1, 1]
        normalized_vel = np.clip(vel ,-8,8)/8
        digitized_rad = np.digitize(normalized_rad, self._obs_space)
        digitized_vel = np.digitize(normalized_vel, self._obs_space)
        return digitized_rad  + digitized_vel * self._config.obs_discrete
    
    def sample_action(self):
        action = np.zeros(self._config.act_discrete)
        index = np.random.randint(0, self._config.act_discrete)
        action[index] = 1
        return action
