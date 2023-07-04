import torch
import numpy as np
from env import DMEnv
from model import DQN
from replay import ReplayBuffer
from dm_control import suite
import dataclasses
import os
from PIL import Image
from torch import nn
from itertools import count
from log import Logger
from collections import OrderedDict
import time

@dataclasses.dataclass
class EnvConfig:
    domain:str ='pendulum'
    task: str = 'swingup'
    act_discrete = 4
    obs_discrete = 16

@dataclasses.dataclass
class Args:
    lr:float = 1e-4
    replay_capacity:int = int(10e6)
    start_step:int = 2000
    batch_size:int = 128
    # horizon:int = 8
    gamma:float = 0.99
    eps_start:float = 0.9
    eps_end:float = 0.05
    eps_decay:float = 8000
    tau:float = 0.005
    episodes:int = int(10e4)
    restore: bool = False
    episode_length: int = 200
    should_update_target: int = 500


class Agent:

    def __init__(self, args:Args, obs_spec, act_spec):
        self.args = args
        self.obs_spec = obs_spec
        self.act_spec = act_spec
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.memory = ReplayBuffer(args.replay_capacity, obs_spec, act_spec, args.batch_size)
        self._build_model()
        if args.restore:
            self._restore()

    def collect_random_episodes(self, env:DMEnv, logger:Logger, episodes:int):
        total_ep_reward = []
        for i in range(episodes):
            ep_reward = 0
            state, _, _, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            for step in range(self.args.episode_length):
                action = torch.tensor(env.sample_action()[None, ...], device=self.device, dtype=torch.long).max(1)[1]
                obs, reward, terminated, truncated = env.step(action)
                ep_reward += reward
                done = terminated or truncated

                if step == self.args.episode_length - 1:
                    done = True

                next_state = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                
                self.memory.add(logger.global_step, state, action.unsqueeze(0), reward, next_state, done)

                state = next_state
                logger.step()

                if done:
                    break
            total_ep_reward.append(ep_reward)
        return np.array(total_ep_reward)

    def collect_one_step(self, env:DMEnv, state, step_in_ep: int, logger:Logger):
        action = self.select_action(env, state, logger.global_step)
        obs, reward, terminated, truncated = env.step(action)
        done = terminated or truncated
        if step_in_ep == self.args.episode_length - 1:
            done = True

        next_state = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

        self.memory.add(logger.global_step, state, action.unsqueeze(0), reward, next_state, done)
        next_state = state

        return next_state, reward, done


    def collect_one_episode(self, env:DMEnv, logger:Logger):
        ep_reward = 0
        state, _, _, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        for i in range(self.args.episode_length):
            action = self.select_action(env, state, logger.global_step)
            obs, reward, terminated, truncated = env.step(action)
            ep_reward += reward
            done = terminated or truncated

            if i == self.args.episode_length - 1:
                done = True

            next_state = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            
            self.memory.add(logger.global_step, state, action.unsqueeze(0), reward, next_state, done)
            logger.step()

            state = next_state

            if done:
                break

        return ep_reward

    def loss(self, non_final_indices, next_state, state, action, reward):
        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state).gather(1, action)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.args.batch_size, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_indices] = self.target_net(next_state[non_final_indices]).max(1)[0]
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.args.gamma) + reward

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        return loss

    def train_one_batch(self):
        if self.memory.idx < self.args.batch_size:
            return
        state,action,reward,next_state,done = self.memory.sample()
        state = torch.tensor(state, dtype=torch.float32, device=self.device)
        action = torch.tensor(action, dtype=torch.long, device=self.device)
        done  = torch.tensor(done, dtype=torch.float32, device=self.device)
        reward = torch.tensor(reward, dtype=torch.float32, device=self.device)
        next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device)
        non_final_indices = torch.where(done == False)[0]
        loss = self.loss(non_final_indices, next_state, state, 
                        action, reward)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 100)
        self.optimizer.step()
        return loss.item()

    def update_target_net(self):
            target_net_state_dict = self.target_net.state_dict()
            policy_net_state_dict = self.policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*self.args.tau + target_net_state_dict[key]*(1-self.args.tau)
            self.target_net.load_state_dict(target_net_state_dict)

    def select_action(self, env, state, global_step):
        if self._scheduler(global_step):
            with torch.no_grad():
                return self.policy_net(state).max(1)[1]
        else:
            return torch.tensor(env.sample_action()[None, ...], device=self.device, dtype=torch.long).max(1)[1]
        
    def _scheduler(self, global_step) -> bool:
        eps_threshold = self.args.eps_end + (self.args.eps_start - self.args.eps_end) * \
            np.exp(-1. * global_step / self.args.eps_decay)
        if np.random.random() > eps_threshold:
            return True
        else:
            return False

    def _build_model(self):
        self.policy_net = DQN(n_observations=self.obs_spec, n_actions=self.act_spec).to(self.device)
        self.target_net = DQN(n_observations=self.obs_spec, n_actions=self.act_spec).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = torch.optim.AdamW(self.policy_net.parameters(), lr=self.args.lr, amsgrad=True)

    def _save_model(self):
        pass

    def _restore(self):
        pass
        # self.policy_net.load_state_dict(torch.load('policy_net.pth'))
        # self.target_net.load_state_dict(torch.load('target_net.pth'))
        # self.optimizer.load_state_dict(torch.load('optimizer.pth'))
    
def main():
    envconfig = EnvConfig()
    args = Args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = DMEnv(envconfig)
    logdir = f"./logs/{time.strftime('%Y-%m-%d-%H-%M-%S')}"
    logger = Logger(logdir)
    agent = Agent(args, env.obs_spec, env.act_spec)

    collect_random_ep = np.ceil(args.start_step // args.episode_length).astype(np.int32)
    initial_ep_reward = agent.collect_random_episodes(env, logger, collect_random_ep)
    print(f"initial_ep_reward: {initial_ep_reward}")

    for i_episode in range(args.episodes):
        # Initialize the environment and get it's state
        ep_reward = 0
        state, _, _, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        for t in count():
            state, reward, done = agent.collect_one_step(env, state, t, logger)
            ep_reward += reward

            # Perform one step of the optimization (on the policy network)
            if (logger.global_step + 1) % 100 == 0:
                loss = agent.train_one_batch()
                logs = OrderedDict()
                logs.update({'loss': loss})
                # print(f"loss: {loss}")
                logger.add_scalars(logs)
            
            if (logger.global_step + 1) % 16 == 0:
                agent.update_target_net()

            logger.step()

            if done:
                break
        if (i_episode + 1) % 1000 == 0:
            img_arr = []
            state, _, _, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            img_arr.append(env._env.physics.render(height=480, width=640, camera_id=0))
            for i in range(200):
                action = agent.select_action(env, state, logger.global_step)
                state, reward, terminated, truncated = env.step(action)
                img_arr.append(env._env.physics.render(height=480, width=640,camera_id=0))
                state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                # logger.add_scalars({'eval_reward': reward}, i)
                # logger.add_scalars({'eval_action': action}, i)
            logger.log_video(img_arr)
        logs = OrderedDict()
        logs.update({'reward': ep_reward})
        logger.add_scalars(logs)
        if i_episode % 100 == 0:
            print(f"\n############## {logger.global_step} #################")
            print(f"Episode {i_episode} finished after {t+1} timesteps")
            print(f"Episode reward: {ep_reward}")
            print("######################################################\n")

    img_arr = []
    state, _, _, _ = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    img_arr.append(env._env.physics.render(height=480, width=640, camera_id=0))
    for i in range(200):
        action = agent.select_action(env, state, logger.global_step)
        state, reward, terminated, truncated = env.step(action)
        img_arr.append(env._env.physics.render(height=480, width=640,camera_id=0))
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        # logger.add_scalars({'eval_reward': reward}, i)
        # logger.add_scalars({'eval_action': action}, i)
    logger.log_video(img_arr)

if __name__ == "__main__":
    main()