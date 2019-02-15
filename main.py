import gym
import torch
import pickle
from model import DeepQNetwork, Agent
import torch.multiprocessing as mp
import numpy as np
import time

import threading

brains = []
scores = []
epsHistory = []
numGames = 1
batch_size = 8

def play(brain, id):
    print('test')
    numGames = 1
    for i in range(numGames):
        #for j in tqdm():
        print('ID : ', id, ',starting game ', i+1, 'epsilon: %.4f' % brain.EPSILON)
        epsHistory.append(brain.EPSILON)
        done = False
        env = gym.make('SpaceInvaders-v0')
        observation = env.reset()
        frames = [np.sum(observation[15:200,30:125], axis=2)]
        score = 0
        lastAction = 0
        delta = 0
        while not done:
            if len(frames) == 3:
                action = brain.chooseAction(frames)
                frames = []
                current_time = time.clock()
                print('ID: ', id, ' - ', current_time - delta)
                delta = current_time
            else:
                action = lastAction
            #env.render()
            observation_, reward, done, info = env.step(action)
            score += reward
            frames.append(np.sum(observation_[15:200,30:125], axis=2))
            if done and info['ale.lives'] == 0:
                reward = -100
            brain.storeTransition(np.mean(observation[15:200,30:125], axis=2), action, reward,
                                  np.mean(observation_[15:200,30:125], axis=2))
            observation = observation_

            #thread = threading.Thread(target=brain.learn, args=batch_size)
            #pbar = tqdm.tqdm(total=10)
            #thread.start()
            #while thread.is_alive():
            #    thread.join(timeout=0.1)
            #    pbar.update(0.1)
            brain.learn(batch_size)
            lastAction = action
    print(score)
    env.close()

if __name__ == '__main__':
    brain = Agent(gamma=0.95, epsilon=0.05,
                  alpha=0.003, maxMemorySize=1000,
                  replace=None)
    env = gym.make('SpaceInvaders-v0')
    observation = env.reset()
    #for i in range(300):
    env.render()
    env.close()
#
    brain.Q_eval.load_state_dict(torch.load("/home/plicca/code/qlearn-academy/qeval.pth"))
    brain.Q_next.load_state_dict(torch.load("/home/plicca/code/qlearn-academy/qnext.pth"))
    brain.memory = pickle.load(open("/home/plicca/code/qlearn-academy/mem.npy", 'rb'))
    # torch.save(brain.Q_eval.state_dict(), "/home/plicca/PycharmProjects/sp/qeval")
    # torch.save(brain.Q_next.state_dict(), "/home/plicca/PycharmProjects/sp/qnext")
    # print("Brain saved");

    # while brain.memCntr < brain.memSize:
    #      observation = env.reset()
    #      done = False
    #      while not done:
    #        #0 no action, 1 fire, 2 move right, 3 move left, 4 move right fire, 5 move left fire
             # env.render()
             # action = env.action_space.sample()
             # observation_, reward, done, info = env.step(action)
             # if done and info['ale.lives'] == 0:
             #     reward = -100
             # brain.storeTransition(np.mean(observation[15:200,30:125], axis=2), action, reward,
             #                     np.mean(observation_[15:200,30:125], axis=2))
             # observation = observation_
    # pickle.dump(brain.memory, open("/home/plicca/PycharmProjects/sp/mem.npy", 'wb'))
    # print("Memory saved")
    # print('done initializing memory')
    #env.close()
    mp.set_start_method('spawn')
    scores = []
    epsHistory = []
    numGames = 1
    batch_size = 16
    # uncomment the line below to record every episode.
    # env = wrappers.Monitor(env, "tmp/space-invaders-1", video_callable=lambda episode_id: True, force=True)
    #env = gym.make('SpaceInvaders-v0')
    #env.close()
    num_processes = 2
    brain.Q_next.share_memory()
    brain.Q_eval.share_memory()
    processes = []
    id = 1
    for rank in range (num_processes):
        p = mp.Process(target=play, args=(brain, id))
        p.start()
        processes.append(p)
        id += 1
    for p in processes:
        p.join()

    # [p.apply(play, args=()) for i in range(0, 1)]

    #for i in range(numGames):
#
    #    for i in range(multiprocessing.cpu_count()):
    #        #for j in tqdm():
    #        print('starting game ', i+1, 'epsilon: %.4f' % brain.EPSILON)
    #        epsHistory.append(brain.EPSILON)
    #        done = False
    #        env = gym.make('SpaceInvaders-v0')
    #        observation = env.reset()
    #        frames = [np.sum(observation[15:200,30:125], axis=2)]
    #        score = 0
    #        lastAction = 0
    #        delta = 0
    #        while not done:
    #            if len(frames) == 3:
    #                action = brain.chooseAction(frames)
    #                frames = []
    #                current_time = time.clock()
    #                print(current_time - delta)
    #                delta = current_time
    #            else:
    #                action = lastAction
    #            env.render()
    #            observation_, reward, done, info = env.step(action)
    #            score += reward
    #            frames.append(np.sum(observation_[15:200,30:125], axis=2))
    #            if done and info['ale.lives'] == 0:
    #                reward = -100
    #            brain.storeTransition(np.mean(observation[15:200,30:125], axis=2), action, reward,
    #                                  np.mean(observation_[15:200,30:125], axis=2))
    #            observation = observation_
#
    #            #thread = threading.Thread(target=brain.learn, args=batch_size)
    #            #pbar = tqdm.tqdm(total=10)
    #            #thread.start()
    #            #while thread.is_alive():
    #            #    thread.join(timeout=0.1)
    #            #    pbar.update(0.1)
    #            brain.learn(batch_size)
    #            lastAction = action
    #    print(score)
    #    env.close()