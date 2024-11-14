import pygame
import sys
import time
import random
import numpy as np

# Q-learning compatible environment

# Constants
FRAME_SIZE_X = 720
FRAME_SIZE_Y = 480
BLOCK_SIZE = 10
FPS = 60

# Game Colors
BLACK = pygame.Color(0, 0, 0)
WHITE = pygame.Color(255, 255, 255)
RED = pygame.Color(255, 0, 0)
GREEN = pygame.Color(0, 255, 0)
BLUE = pygame.Color(0, 0, 255)

class SnakeEnv:
    def __init__(self):
        # Initialize PyGame
        pygame.init()
        pygame.display.set_caption("Snake Eater")
        self.game_window = pygame.display.set_mode((FRAME_SIZE_X, FRAME_SIZE_Y))
        
        # Initialize game variables
        self.snake_pos = [100, 50]
        self.snake_body = [[100, 50], [90, 50], [80, 50]]
        self.food_pos = self.spawn_food()
        self.food_spawn = True
        self.direction = 'RIGHT'
        self.score = 0

    def spawn_food(self):
        return [random.randrange(1, FRAME_SIZE_X // BLOCK_SIZE) * BLOCK_SIZE, 
                random.randrange(1, FRAME_SIZE_Y // BLOCK_SIZE) * BLOCK_SIZE]

    def reset(self):
        """Resets the game to start a new episode"""
        self.snake_pos = [100, 50]
        self.snake_body = [[100, 50], [90, 50], [80, 50]]
        self.food_pos = self.spawn_food()
        self.food_spawn = True
        self.direction = 'RIGHT'
        self.score = 0
        return self.get_state()
    
    def get_state(self):
        """Returns the current state as a list of important variables for Q-learning"""
        # State includes snake head position, food position, and direction
        return [
            self.snake_pos[0], self.snake_pos[1],
            self.food_pos[0], self.food_pos[1],
            int(self.direction == 'UP'), int(self.direction == 'DOWN'), 
            int(self.direction == 'LEFT'), int(self.direction == 'RIGHT')
        ]

    def step(self, action):
        """
        Takes an action and updates the environment, returning next_state, reward, and done
        Actions: 0 - UP, 1 - DOWN, 2 - LEFT, 3 - RIGHT
        """
        # Update direction based on action
        directions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        new_direction = directions[action]
        
        # Prevent the snake from moving in the opposite direction instantly
        if (new_direction == 'UP' and self.direction != 'DOWN') or \
           (new_direction == 'DOWN' and self.direction != 'UP') or \
           (new_direction == 'LEFT' and self.direction != 'RIGHT') or \
           (new_direction == 'RIGHT' and self.direction != 'LEFT'):
            self.direction = new_direction

        # Move the snake
        if self.direction == 'UP':
            self.snake_pos[1] -= BLOCK_SIZE
        elif self.direction == 'DOWN':
            self.snake_pos[1] += BLOCK_SIZE
        elif self.direction == 'LEFT':
            self.snake_pos[0] -= BLOCK_SIZE
        elif self.direction == 'RIGHT':
            self.snake_pos[0] += BLOCK_SIZE

        # Growing the snake
        self.snake_body.insert(0, list(self.snake_pos))
        
        # Check if food is eaten
        reward = 0
        done = False
        if self.snake_pos == self.food_pos:
            self.score += 1
            reward = 10  # Positive reward for eating food
            self.food_spawn = False
        else:
            self.snake_body.pop()

        # Spawn new food if needed
        if not self.food_spawn:
            self.food_pos = self.spawn_food()
        self.food_spawn = True

        # Check for game over conditions
        if self.is_collision():
            reward = -10  # Negative reward for collision
            done = True
        
        # Get the next state
        next_state = self.get_state()
        return next_state, reward, done

    def is_collision(self):
        # Check for wall collision
        if (self.snake_pos[0] < 0 or self.snake_pos[0] >= FRAME_SIZE_X or
            self.snake_pos[1] < 0 or self.snake_pos[1] >= FRAME_SIZE_Y):
            return True
        # Check for collision with itself
        if self.snake_pos in self.snake_body[1:]:
            return True
        return False

    def render(self):
        """Renders the game for visualizing training"""
        self.game_window.fill(BLACK)
        for pos in self.snake_body:
            pygame.draw.rect(self.game_window, GREEN, pygame.Rect(pos[0], pos[1], BLOCK_SIZE, BLOCK_SIZE))
        pygame.draw.rect(self.game_window, WHITE, pygame.Rect(self.food_pos[0], self.food_pos[1], BLOCK_SIZE, BLOCK_SIZE))
        pygame.display.flip()
        pygame.time.Clock().tick(FPS)

    def close(self):
        pygame.quit()
        sys.exit()
