import numpy as np
from snake_game import Env
import matplotlib.pyplot as plt
from matplotlib import style
import time
import pickle

FOOD_REWARD = 10
BLOCK_SIZE = 10
SIZE_X = 720
SIZE_Y = 480

style.use('ggplot')

LEARNING_RATE = 0.1
DISCOUNT = 0.9

epsilon = 0.8
EPS_DECAY = 0.998

EPISODES = 25000
SHOW_EVERY  = 1000
MOVES_PER_EP = 200

start_q_table = None

env = Env()

if start_q_table is None:
    q_table = {}

    for x1 in range(-SIZE_X+1,SIZE_X):
        for y1 in range(-SIZE_Y+1,SIZE_Y):
            for x2 in range(-SIZE_X+1,SIZE_X):
                for y2 in range(-SIZE_Y+1,SIZE_Y):
                    q_table[((x1,y1),((x2,y2)))] = [np.random.uniform(-5,0) for i in range(4)]
else:
    with open(start_q_table,'rb') as f:
        q_table = pickle.load(f)

epsiode_rewards = []

def discrete_state(state):
    snake_pos = [state[0],state[1]]
    food_pos = [state[2],state[3]]

    direction = [state[4],state[5],state[6],state[7]]

    discrete_snake_pos = (snake_pos[0]//BLOCK_SIZE,snake_pos[1]//BLOCK_SIZE)
    discrete_food_pos = (food_pos[0]//BLOCK_SIZE,food_pos[1]//BLOCK_SIZE)

    discrete = (discrete_snake_pos,discrete_food_pos,direction)
    return discrete

for episode in range(EPISODES):

    if episode%SHOW_EVERY == 0:
        print(f"on # {episode}, epsilon: {epsilon}")
        print(f"{SHOW_EVERY} ep mean {np.mean(epsiode_rewards[-SHOW_EVERY:])}")
        show = True
    else:
        show = False
    
    epsiode_reward = 0

    state = discrete_state(env.reset())

    for i in range(MOVES_PER_EP):

        if np.random.rand() > epsilon:
            action = np.argmax(q_table[state])
        else:
            action = np.random.randint(0,4)

        new_state,reward,done = env.step(action)

        new_discrete_state = discrete_state(new_state)

        max_future_q = np.max(q_table[new_discrete_state])
        curr_q = q_table[state][action]

        if reward == FOOD_REWARD:
            new_q = FOOD_REWARD
        else:
            new_q = (1-LEARNING_RATE)*curr_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)

        q_table[state][action] = new_q

        if show:
            env.render()
            time.sleep(0.05)
        
        epsiode_reward+= reward

        if done:
            break
    
    epsiode_rewards.append(epsiode_reward)
    epsilon *= EPS_DECAY

moving_avg = np.convolve(epsiode_rewards,np.ones((SHOW_EVERY,))//SHOW_EVERY,mode='valid')

plt.plot([i for i in range(len(moving_avg))],moving_avg)
plt.ylabel(f"Reward {SHOW_EVERY}")
plt.xlabel("Episode #")
plt.show()

with open(f"q-table{int(time.time())}.pickle",'wb') as f:
    pickle.dump(q_table,f)

env.close()