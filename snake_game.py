import pygame
import sys
import time
import random
# from enum import Enum
import numpy as np
import matplotlib.pyplot as plt
import pickle
from matplotlib import style

SCALING_FACTOR = 32
SIZE_X = 12
SIZE_Y = 12
FPS = 60

BOUNDARY_PENALTY = 300
SELF_PENALTY = 150
FOOD_REWARD = 25

#Colors
BLACK = pygame.Color(0,0,0)
WHITE = pygame.Color(255,255,255)
RED = pygame.Color(255,0,0)
GREEN = pygame.Color(0,255,0)
BLUE = pygame.Color(0,0,255)

class Env:
    def __init__(self):
        pygame.init()

        self.clock = pygame.time.Clock()
        
        pygame.display.set_caption("Snake Game")

        self.game_window = pygame.display.set_mode((SIZE_X*SCALING_FACTOR,SIZE_Y*SCALING_FACTOR))

        self.snake_pos = [2, 2]
        self.snake_body = [[2,2],[1,2]]
        self.food_pos = self._spawnFood()
        self.food_spawn = True
        self.direction = 'RIGHT'
        self.score = 0

    def _spawnFood(self):
        return [random.randrange(1,SIZE_X),random.randrange(1,SIZE_Y)]
    
    def get_action_size(self):
        return 4

    def reset(self):
        self.snake_pos = [2,2]
        self.snake_body = [[2,2],[1,2]]
        self.food_pos = self._spawnFood()
        self.food_spawn = True
        self.direction = 'RIGHT'
        self.score = 0

        done = False

        return self.get_state()
    
    def get_rewards(self):
        return [FOOD_REWARD,BOUNDARY_PENALTY,SELF_PENALTY]

    def get_state(self):
        return ((self.snake_pos[0],self.snake_pos[1]),
                (self.food_pos[0],self.food_pos[1]))
    
    def step(self,action):
        directions = ['UP','DOWN','LEFT','RIGHT']
        new_direction = directions[action]

        #To stop snake from moving in opposite direction immediately
        if (new_direction == 'UP' and self.direction != 'DOWN') or \
           (new_direction == 'DOWN' and self.direction != 'UP') or \
           (new_direction == 'LEFT' and self.direction != 'RIGHT') or \
           (new_direction == 'RIGHT' and self.direction != 'LEFT'):
            self.direction = new_direction

        old_dist = abs(self.snake_pos[0] - self.food_pos[0]) + abs(self.snake_pos[1] - self.food_pos[1])
        
        #Move
        if self.direction == 'UP':
            self.snake_pos[1] -= 1
        elif self.direction == 'DOWN':
            self.snake_pos[1] += 1
        elif self.direction == 'LEFT':
            self.snake_pos[0] -= 1
        elif self.direction == 'RIGHT':
            self.snake_pos[0] += 1

        new_dist = abs(self.snake_pos[0] - self.food_pos[0]) + abs(self.snake_pos[1] - self.food_pos[1])
    
        reward = old_dist - new_dist
        done = False

        self.snake_body.insert(0,list(self.snake_pos))

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
            done = True
            reward = -SELF_PENALTY
        
        new_state = self.get_state()
        return new_state,reward,done
    
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
            pygame.draw.rect(self.game_window,GREEN,pygame.Rect(pos[0]*SCALING_FACTOR,pos[1]*SCALING_FACTOR,SCALING_FACTOR,SCALING_FACTOR))
        
        pygame.draw.rect(self.game_window,WHITE,pygame.Rect(self.food_pos[0]*SCALING_FACTOR,self.food_pos[1]*SCALING_FACTOR,SCALING_FACTOR,SCALING_FACTOR))
        pygame.display.flip()
        self.clock.tick(FPS)

    def close(self):
        pygame.quit()
        sys.exit()
