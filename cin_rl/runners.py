from collections import namedtuple
import numpy as np
import itertools

Experience = namedtuple("Experience", ["obs", "action", "reward", "newobs", "done"])

def episode_runner(envfn, f, render=False):
    env = envfn()
    while True:
        history = []
        done = False
        obs = env.reset()
        while not done:
            render and env.render()
            action = f(obs)
            nobs, reward, done, _ = env.step(action)
            history.append(Experience(obs, action, reward, nobs, done))
            obs = nobs
        render and env.render()
        yield Experience(*map(np.array, zip(*history)))
        del history[:]

def run_single(envfn, f):
    return next(episode_runner(envfn, f))

def nstep_runner(envfn, f, steps):
    env = envfn()
    history = []
    done = False
    obs = env.reset()
    
    obs_history = np.zeros((steps, *obs.shape))
    nobs_history = np.zeros_like(obs_history)
    reward_history = np.zeros(steps)
    action_history = np.zeros(steps)
    done_history = np.ones(steps)

    while True:
        for timestep in range(steps):
            action = f(obs)
            nobs, reward, done, _ = env.step(action)
            action_history[timestep] = action
            reward_history[timestep] = reward
            done_history[timestep] = done
            nobs_history[timestep] = nobs
            obs_history[timestep] = obs
            obs = nobs
            if done:
                obs = env.reset()
                done = False
                break
        
        yield Experience(obs_history, action_history, reward_history, nobs_history, done_history)
        obs_history = np.zeros((steps, *obs.shape))
        nobs_history = np.zeros_like(obs_history)
        reward_history = np.zeros(steps)
        action_history = np.zeros(steps)
        done_history = np.ones(steps)