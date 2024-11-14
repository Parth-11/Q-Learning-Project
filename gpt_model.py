import pygame
import sys
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import pickle
from matplotlib import style

SIZE_X = 720
SIZE_Y = 480
BLOCK_SIZE = 10
FPS = 60

BOUNDARY_PENALTY = 100
SELF_PENALTY = 150
FOOD_REWARD = 10

BLACK = pygame.Color(0, 0, 0)
WHITE = pygame.Color(255, 255, 255)
RED = pygame.Color(255, 0, 0)
GREEN = pygame.Color(0, 255, 0)

class Env:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Snake Game")
        self.clock = pygame.time.Clock()
        self.game_window = pygame.display.set_mode((SIZE_X, SIZE_Y))
        self.snake_pos = [100, 50]
        self.snake_body = [[100, 50], [90, 50], [80, 50]]
        self.food_pos = self._spawnFood()
        self.food_spawn = True
        self.direction = 'RIGHT'
        self.score = 0

    def _spawnFood(self):
        return [random.randrange(1, SIZE_X // BLOCK_SIZE) * BLOCK_SIZE,
                random.randrange(1, SIZE_Y // BLOCK_SIZE) * BLOCK_SIZE]
    
    def get_action_size(self):
        return 4

    def reset(self):
        self.snake_pos = [100, 50]
        self.snake_body = [[100, 50], [90, 50], [80, 50]]
        self.food_pos = self._spawnFood()
        self.food_spawn = True
        self.direction = 'RIGHT'
        self.score = 0
        return self.get_state()
    
    def get_state(self):
        return [self.snake_pos[0], self.snake_pos[1],
                self.food_pos[0], self.food_pos[1],
                int(self.direction == 'UP'), int(self.direction == 'DOWN'),
                int(self.direction == 'LEFT'), int(self.direction == 'RIGHT')]
    
    def step(self, action):
        directions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        new_direction = directions[action]

        if (new_direction == 'UP' and self.direction != 'DOWN') or \
           (new_direction == 'DOWN' and self.direction != 'UP') or \
           (new_direction == 'LEFT' and self.direction != 'RIGHT') or \
           (new_direction == 'RIGHT' and self.direction != 'LEFT'):
            self.direction = new_direction

        if self.direction == 'UP':
            self.snake_pos[1] -= BLOCK_SIZE
        elif self.direction == 'DOWN':
            self.snake_pos[1] += BLOCK_SIZE
        elif self.direction == 'LEFT':
            self.snake_pos[0] -= BLOCK_SIZE
        elif self.direction == 'RIGHT':
            self.snake_pos[0] += BLOCK_SIZE
    
        reward = 0
        done = False

        self.snake_body.insert(0, list(self.snake_pos))

        if self.snake_pos == self.food_pos:
            self.score += 1
            reward = FOOD_REWARD
            self.food_spawn = False
        else:
            self.snake_body.pop()

        if not self.food_spawn:
            self.food_pos = self._spawnFood()
            self.food_spawn = True

        if self._isCollisionBoundary():
            reward = -BOUNDARY_PENALTY
            done = True

        if self._isCollisionSelf():
            reward = -SELF_PENALTY
            done = True

        new_state = self.get_state()
        return new_state, reward, done
    
    def _isCollisionBoundary(self):
        if (self.snake_pos[0] < 0 or self.snake_pos[0] >= SIZE_X
                or self.snake_pos[1] < 0 or self.snake_pos[1] >= SIZE_Y):
            return True
        return False

    def _isCollisionSelf(self):
        if self.snake_pos in self.snake_body[1:]:
            return True
        return False
    
    def render(self):
        self.game_window.fill(BLACK)
        for pos in self.snake_body:
            pygame.draw.rect(self.game_window, GREEN, pygame.Rect(pos[0], pos[1], BLOCK_SIZE, BLOCK_SIZE))
        
        pygame.draw.rect(self.game_window, WHITE, pygame.Rect(self.food_pos[0], self.food_pos[1], BLOCK_SIZE, BLOCK_SIZE))
        pygame.display.flip()
        self.clock.tick(FPS)

    def close(self):
        pygame.quit()
        sys.exit()


##Q Learning

style.use('ggplot')

LEARNING_RATE = 0.1
DISCOUNT = 0.9

epsilon = 0.8
EPS_DECAY = 0.998

EPISODES = 25000
SHOW_EVERY = 1000
MOVES_PER_EP = 200

start_q_table = None

env = Env()

if start_q_table is None:
    q_table = {}
    for x1 in range(-SIZE_X // BLOCK_SIZE, SIZE_X // BLOCK_SIZE):
        for y1 in range(-SIZE_Y // BLOCK_SIZE, SIZE_Y // BLOCK_SIZE):
            for x2 in range(-SIZE_X // BLOCK_SIZE, SIZE_X // BLOCK_SIZE):
                for y2 in range(-SIZE_Y // BLOCK_SIZE, SIZE_Y // BLOCK_SIZE):
                    q_table[((x1, y1), (x2, y2))] = [np.random.uniform(-5, 0) for _ in range(4)]
else:
    with open(start_q_table, 'rb') as f:
        q_table = pickle.load(f)

episode_rewards = []

def discrete_state(state):
    snake_pos = (state[0] // BLOCK_SIZE, state[1] // BLOCK_SIZE)
    food_pos = (state[2] // BLOCK_SIZE, state[3] // BLOCK_SIZE)
    direction = tuple(state[4:8])
    return (snake_pos, food_pos, direction)

for episode in range(EPISODES):

    if episode % SHOW_EVERY == 0:
        print(f"on # {episode}, epsilon: {epsilon}")
        print(f"{SHOW_EVERY} ep mean {np.mean(episode_rewards[-SHOW_EVERY:])}")
        show = True
    else:
        show = False

    episode_reward = 0
    state = discrete_state(env.reset())

    for _ in range(MOVES_PER_EP):

        if state not in q_table:
            q_table[state] = [np.random.uniform(-5, 0) for _ in range(4)]

        if np.random.rand() > epsilon:
            action = np.argmax(q_table[state])
        else:
            action = np.random.randint(0, 4)

        new_state, reward, done = env.step(action)
        new_discrete_state = discrete_state(new_state)

        if new_discrete_state not in q_table:
            q_table[new_discrete_state] = [np.random.uniform(-5, 0) for _ in range(4)]

        max_future_q = np.max(q_table[new_discrete_state])
        current_q = q_table[state][action]

        if reward == FOOD_REWARD:
            new_q = FOOD_REWARD
        else:
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)

        q_table[state][action] = new_q

        if show:
            env.render()
            time.sleep(0.05)
        
        episode_reward += reward

        if done:
            break

        state = new_discrete_state

    episode_rewards.append(episode_reward)
    epsilon *= EPS_DECAY

moving_avg = np.convolve(episode_rewards, np.ones(SHOW_EVERY) / SHOW_EVERY, mode='valid')

plt.plot([i for i in range(len(moving_avg))], moving_avg)
plt.ylabel(f"Reward {SHOW_EVERY}")
plt.xlabel("Episode #")
plt.show()

with open(f"q-table-{int(time.time())}.pickle", 'wb') as f:
    pickle.dump(q_table, f)

env.close()
