import pygame, sys, random
import numpy as np

# Initialize PyGame
pygame.init()

# Difficulty settings
difficulty = 10  # Adjust as needed for testing speed

# Window size
frame_size_x = 720
frame_size_y = 480

# Colors
black = pygame.Color(0, 0, 0)
white = pygame.Color(255, 255, 255)
red = pygame.Color(255, 0, 0)
green = pygame.Color(0, 255, 0)

# Game Display
pygame.display.set_caption('Snake Eater AI')
game_window = pygame.display.set_mode((frame_size_x, frame_size_y))

# FPS controller
fps_controller = pygame.time.Clock()

# Q-learning Parameters
actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
q_table = {}
learning_rate = 0.1
discount_factor = 0.9
epsilon = 1.0  # Exploration rate
epsilon_decay = 0.995
min_epsilon = 0.01

# Game Variables
def reset_game():
    global snake_pos, snake_body, food_pos, food_spawn, score
    snake_pos = [100, 50]
    snake_body = [[100, 50], [90, 50], [80, 50]]
    food_pos = [random.randrange(1, (frame_size_x//10)) * 10, random.randrange(1, (frame_size_y//10)) * 10]
    food_spawn = True
    score = 0

def get_state():
    danger = [
        (snake_pos[0] == 0 or [snake_pos[0] - 10, snake_pos[1]] in snake_body),  # Danger left
        (snake_pos[0] == frame_size_x - 10 or [snake_pos[0] + 10, snake_pos[1]] in snake_body),  # Danger right
        (snake_pos[1] == 0 or [snake_pos[0], snake_pos[1] - 10] in snake_body),  # Danger up
        (snake_pos[1] == frame_size_y - 10 or [snake_pos[0], snake_pos[1] + 10] in snake_body),  # Danger down
    ]
    food_direction = (food_pos[0] - snake_pos[0], food_pos[1] - snake_pos[1])
    return (*danger, *food_direction)

def choose_action(state):
    if np.random.rand() < epsilon:  # Exploration
        return random.choice(actions)
    if state not in q_table:
        q_table[state] = np.zeros(len(actions))  # Initialize if state is new
    return actions[np.argmax(q_table[state])]

def update_q_table(state, action, reward, next_state):
    if state not in q_table:
        q_table[state] = np.zeros(len(actions))
    if next_state not in q_table:
        q_table[next_state] = np.zeros(len(actions))
    action_index = actions.index(action)
    best_future_q = np.max(q_table[next_state])
    q_table[state][action_index] = q_table[state][action_index] + learning_rate * (reward + discount_factor * best_future_q - q_table[state][action_index])

# Render function for display
def render_game():
    game_window.fill(black)
    for pos in snake_body:
        pygame.draw.rect(game_window, green, pygame.Rect(pos[0], pos[1], 10, 10))
    pygame.draw.rect(game_window, white, pygame.Rect(food_pos[0], food_pos[1], 10, 10))
    pygame.display.update()

# Main Game Loop
for episode in range(100):  # Number of games to train
    reset_game()
    direction = 'RIGHT'
    while True:
        state = get_state()
        action = choose_action(state)

        # Convert action to movement
        if action == 'UP' and direction != 'DOWN':
            direction = 'UP'
        elif action == 'DOWN' and direction != 'UP':
            direction = 'DOWN'
        elif action == 'LEFT' and direction != 'RIGHT':
            direction = 'LEFT'
        elif action == 'RIGHT' and direction != 'LEFT':
            direction = 'RIGHT'

        # Move snake in the chosen direction
        if direction == 'UP':
            snake_pos[1] -= 10
        elif direction == 'DOWN':
            snake_pos[1] += 10
        elif direction == 'LEFT':
            snake_pos[0] -= 10
        elif direction == 'RIGHT':
            snake_pos[0] += 10

        # Snake eats food
        if snake_pos == food_pos:
            score += 1
            food_spawn = False
            reward = 10
        else:
            snake_body.pop()
            reward = -0.1

        # Spawn food
        if not food_spawn:
            food_pos = [random.randrange(1, (frame_size_x//10)) * 10, random.randrange(1, (frame_size_y//10)) * 10]
        food_spawn = True

        # Game Over conditions
        if (snake_pos[0] < 0 or snake_pos[0] >= frame_size_x or
            snake_pos[1] < 0 or snake_pos[1] >= frame_size_y or
            snake_pos in snake_body):
            reward = -10
            update_q_table(state, action, reward, get_state())
            break

        # Update snake
        snake_body.insert(0, list(snake_pos))
        next_state = get_state()
        update_q_table(state, action, reward, next_state)

        # Render the game and control the speed
        render_game()
        fps_controller.tick(difficulty)

        # Reduce epsilon over time
        epsilon = max(min_epsilon, epsilon * epsilon_decay)

# Quit Game
pygame.quit()
sys.exit()
