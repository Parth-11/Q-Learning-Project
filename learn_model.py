import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import pickle
import time
from snake_game import Env

style.use('ggplot')

LEARNING_RATE = 0.1
DISCOUNT = 0.95

epsilon = 0.8
EPS_DECAY = 0.998

EPISODES = 25000
SHOW_EVERY = 1000
MOVES_PER_EP = 200

start_q_table = None

env = Env()

rewards = env.get_rewards()

DISCRETE_OS_SIZE = [10] * len(env.get_state())

if start_q_table is None:
    q_table = np.random.uniform(low=-2,high=0,size=(DISCRETE_OS_SIZE+[env.get_action_size(),]))
else:
    with open(start_q_table,'rb') as f:
        q_table = pickle.load(f)

episode_rewards = []

def discrete_state(state):

    snake_pos = [state[0],state[1]]
    food_pos = [state[2],state[3]]
    direction = [state[4],state[5],state[6],state[7]]

    discrete_snake_pos = (snake_pos[0]//10,snake_pos[1]//10)
    discrete_food_pos = (food_pos[0]//10,food_pos[1]//10)

    discrete_state = (discrete_snake_pos,discrete_food_pos,direction)

    return discrete_state

for episode in range(EPISODES):
    
    if episode%SHOW_EVERY == 0:
        print(f"on # {episode}, epsilon: {epsilon}")
        print(f"{SHOW_EVERY} ep mean {np.mean(episode_rewards[-SHOW_EVERY:])}")
        show = True
    else:
        show = False

    episode_reward = 0

    state = discrete_state(env.reset())

    for i in range(MOVES_PER_EP):

        if np.random.rand() > epsilon:
            action = np.argmax(q_table[state])
        else:
            action = np.random.randint(0,4)

        new_state ,reward,done = env.step(action)

        new_discrete_state = discrete_state(new_state)

        max_future_q = np.max(q_table[new_discrete_state])
        curr_q = q_table[state][action]

        if reward == rewards[0]:
            new_q = rewards[0]
        elif reward == rewards[1]:
            reward = - rewards[1]
        elif reward == rewards[2]:
            reward = - rewards[2]
        else:
            new_q = (1 - LEARNING_RATE) * curr_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
    

        q_table[state][action] = new_q

        if show:
            env.render()
            time.sleep(0.05)
        
        episode_reward+=reward

        if done:
            break
            
    episode_rewards.append(episode_reward)
    epsilon *=  EPS_DECAY

moving_avg = np.convolve(episode_rewards,np.ones((SHOW_EVERY,))//SHOW_EVERY,mode='valid')

plt.plot([i for i in range(len(moving_avg))],moving_avg)
plt.ylabel(f"Reward {SHOW_EVERY}")
plt.xlabel("Episode #")
plt.show()

with open(f"q-table{int(time.time())}.pickle",'wb') as f:
    pickle.dump(q_table,f)

env.close()