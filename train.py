import torch
import numpy as np
from env import make_env
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
    domain: str = "autostop"
    N: int = 6
    time_limit: int = 100


@dataclasses.dataclass
class Args:
    lr:float = 1e-4
    replay_capacity:int = int(10e6)
    start_step:int = 10000
    batch_size:int = 128
    gamma:float = 0.99
    eps_start:float = 0.95
    eps_end:float = 0.05
    eps_decay:float = 8000
    tau:float = 0.005
    episodes:int = int(10e4)
    restore: bool = False
    episode_length: int = 100
    should_update_target: int = 500
    eval_times: int = 100
    eval_interval: int = 1000
    log_interval: int = 100


class Agent:

    def __init__(self, args:Args, obs_spec, act_spec):
        self.args = args
        self.obs_spec = obs_spec
        self.act_spec = act_spec
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.memory = ReplayBuffer(args.replay_capacity, obs_spec, 1, args.batch_size)
        self._build_model()
        if args.restore:
            self._restore()

    def collect_random_episodes(self, env, logger:Logger, collect_step:int):
        total_ep_reward = []
        ep_i = 0
        while True:
            ep_reward = 0
            state, _, _, state_int = env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=self.device)
            for step in range(self.args.episode_length):
                action = torch.tensor(env.sample_action(), device=self.device, dtype=torch.long)
                state, reward, done, state_int = env.step(action)
                ep_reward += reward
                if step == self.args.episode_length - 1:
                    done = True
                next_state = torch.tensor(state, dtype=torch.float32, device=self.device)
                
                self.memory.add(logger.global_step, state, action, reward, next_state, done)
                state = next_state
                logger.step()
                if done:
                    break
            if logger.global_step > collect_step:
                break
            total_ep_reward.append(ep_reward)
        return np.array(total_ep_reward)

    def collect_one_step(self, env, state, step_in_ep: int, logger:Logger):
        action = self.select_action(env, state, logger.global_step)
        obs, reward, done, state_int = env.step(action)
        if step_in_ep == self.args.episode_length - 1:
            done = True

        next_state = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

        self.memory.add(logger.global_step, state, action, reward, next_state, done)

        return next_state, reward, done

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
        next_state_values = torch.zeros(state_action_values.shape[:-1], device=self.device)
        with torch.no_grad():
            next_state_values[non_final_indices] = self.target_net(next_state[non_final_indices]).max(-1)[0]
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

    def select_action(self, env, state, global_step, explore=True):
        if explore:
            if self._scheduler(global_step):
                with torch.no_grad():
                    return self.policy_net(state).max(-1)[1][0]
            else:
                return torch.tensor(env.sample_action(), device=self.device, dtype=torch.long)
        else:
            with torch.no_grad():
                return self.policy_net(state).max(-1)[1][0]

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
    env = make_env(envconfig)
    logdir = f"./logs/{envconfig.domain}/{time.strftime('%Y-%m-%d-%H-%M-%S')}"
    logger = Logger(logdir)
    agent = Agent(args, env.obs_spec, env.act_spec)

    initial_ep_reward = agent.collect_random_episodes(env, logger, args.start_step)
    print(f"initial_ep_reward: {initial_ep_reward}")

    for i_episode in range(args.episodes):
        # Initialize the environment and get it's state
        ep_reward = 0
        t = 0
        state, _, _, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        while True:
            state, reward, done = agent.collect_one_step(env, state, t, logger)
            ep_reward += reward
            # Perform one step of the optimization (on the policy network)
            loss = agent.train_one_batch()
            if logger.global_step % args.log_interval == 0:
                logs = OrderedDict()
                logs.update({'loss': loss})
                logger.add_scalars(logs)
            agent.update_target_net()
            logger.step()
            t += 1
            if done:
                break

        # for train log
        logs = OrderedDict()
        logs.update({'reward': ep_reward})
        logger.add_scalars(logs)
        if i_episode % 100 == 0:
            print(f"\n############## {logger.global_step} #################")
            print(f"Episode {i_episode} finished after {t} timesteps")
            print(f"Episode reward: {ep_reward}")
            print(f"first state: {env.first_state_int}, last state: {env.state_int}, answer: {env.target_int}")
            print("######################################################\n")

        # for eval
        if (i_episode) % args.eval_interval == 0:
            correct_num = 0
            total_steps = 0
            for _ in range(args.eval_times):
                state, _, _, state_int = env.reset()
                state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                for i in range(args.episode_length):
                    action = agent.select_action(env, state, logger.global_step, explore=False)
                    state, reward, done, state_int = env.step(action)
                    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                    if done:
                        if state_int == env.target_int:
                            correct_num += 1
                        total_steps += i
                        break
                    if i == args.episode_length - 1:
                        total_steps += i
            average_steps = total_steps / args.eval_times
            accuracy = correct_num / args.eval_times
            logs = OrderedDict()
            print(f"average_steps: {average_steps}, accuracy: {accuracy}")
            logs.update({'average_steps': average_steps, 'accuracy': accuracy})
            logger.add_scalars(logs, i_episode)

if __name__ == "__main__":
    main()