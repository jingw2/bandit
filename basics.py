#/usr/bin/python 3.7
#-*-coding:utf-8-*-

'''
Bandit Basics
Bandit Gym Enviroment Link:
https://github.com/JKCooper2/gym-bandits
'''

import gym
import gym_bandits
import numpy as np
import random
import math
import matplotlib.pyplot as plt

def explore_first(N, T, env):
    '''
    Explore-first algorithm
    Args:
    N (int): explore N times for each arm
    T (int): total time steps, N * K < T
    '''   
    if not isinstance(env.action_space, gym.spaces.Discrete):
        raise Exception("Please check the action space is \
            gym.spaces.Discrete format!")
    num_arms = env.action_space.n
    if T < N * num_arms:
        raise Exception("Total time step T should be greater than N * K!")
    arm_cnt = {arm: 0 for arm in range(num_arms)}
    arm_reward = {arm: [] for arm in range(num_arms)}
    # exploration
    rewards = []
    for arm in range(num_arms):
        env.reset()
        while arm_cnt[arm] < N:
            next_state, reward, done, _ = env.step(arm)
            rewards.append(reward)
            arm_reward[arm].append(reward)
            arm_cnt[arm] += 1
            if done:
                env.reset()
    # exploitation
    best_arm = sorted(arm_reward.items(), key=lambda x: np.mean(x[1]), reverse=True)[0][0]
    for i in range(T-num_arms*N):
        env.reset()
        next_state, reward, done, _ = env.step(best_arm)
        rewards.append(reward)
        if done:
            env.reset()
    return rewards

def epsilon_greedy(T, env):
    '''
    Epsilon greedy algorithm
    Args:
    T (int): total time steps
    env (gym.Environment)
    '''
    if not isinstance(env.action_space, gym.spaces.Discrete):
        raise Exception("Please check the action space is \
            gym.spaces.Discrete format!")
    K = env.action_space.n
    arm_reward = {arm: [] for arm in range(K)}
    is_success = False
    rewards = []
    for t in range(T):
        env.reset()
        if t > 1:
            epsilon = t ** (-1/3.) * (K * math.log(t)) ** (1/3)
            is_success = (random.random() < epsilon)
        if is_success or t == 0:
            arm = env.action_space.sample()
        else:
            arm = sorted(arm_reward.items(), key=lambda x: np.mean(x[1]), reverse=True)[0][0]
        next_state, reward, done, _ = env.step(arm)
        rewards.append(reward)
        arm_reward[arm].append(reward)
        if done:
            env.reset()
    return rewards
        
def successive_elimination(N, T, env, ai):
    '''
    Successive elimination
    Args:
    T (int): total time steps
    env (gym.Environment)
    ai (int): alternative_interval
    '''
    if not isinstance(env.action_space, gym.spaces.Discrete):
        raise Exception("Please check the action space is \
            gym.spaces.Discrete format!")
    K = env.action_space.n
    arm_reward = {arm: [] for arm in range(K)}
    eliminated = []
    rewards = []
    env.reset()
    t = 0
    while True:
        if t != 0 and t % ai == 0:
            ucb, lcb = get_cb(arm_reward, T, eliminated)
            if len(ucb) == 1:
                best_arm = list(ucb.keys())[0]
                break
            else:
                for arm1 in ucb:
                    for arm2 in lcb:
                        if ucb[arm1] < lcb[arm2]:
                            eliminated.append(arm1)
                available_arm = [arm for arm in range(K) 
                    if arm not in eliminated]
                arm = random.choice(available_arm)
        else:
            arm = env.action_space.sample()
        next_state, reward, done, _ = env.step(arm)
        rewards.append(reward)
        arm_reward[arm].append(reward)
        t += 1
        if done:
            env.reset()
    for i in range(T-K*N):
        env.reset()
        next_state, reward, done, _ = env.step(best_arm)
        rewards.append(reward)
        if done:
            env.reset()
    return rewards

def get_cb(arm_reward, T, eliminated):
    ucb = {}
    lcb = {}
    for arm, rewards in arm_reward.items():
        if arm in eliminated: continue
        n = len(rewards)
        if n != 0:
            r = math.sqrt(math.log(T) * 2 / float(n))
            mu = np.mean(rewards)
            ucb[arm] = mu + r 
            lcb[arm] = mu - r
    return ucb, lcb
            

if __name__ == "__main__":
    env = gym.make("BanditTenArmedUniformDistributedReward-v0")
    N = 1000
    T = 30000
    ai = 2000
    # rewards = epsilon_greedy(T, env)
    # plt.plot(rewards)
    # rewards = explore_first(N, T, env)
    rewards = successive_elimination(N, T, env, ai)
