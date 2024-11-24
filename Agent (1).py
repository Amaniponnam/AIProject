import pygame
import numpy as np
import random
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Initialize pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 300, 300  # Screen dimensions
GRID_SIZE = 10            # Size of each grid cell
FPS = 15                  # Frames per second

# Colors
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLACK = (0, 0, 0)

# Directions
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

# Q-learning Hyperparameters
ALPHA = 0.1        # Learning rate
GAMMA = 0.9        # Discount factor
EPSILON_START = 1  # Initial exploration probability
EPSILON_END = 0.1  # Final exploration probability
EPSILON_DECAY = 0.995  # Decay rate for epsilon
EPISODES = 1000

# Pygame setup
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Snake Game")
clock = pygame.time.Clock()


class SnakeGame:
    def __init__(self):
        self.reset()

    def reset(self):
        """Resets the game state."""
        self.snake = [(5, 5)]
        self.direction = RIGHT
        self.food = self.spawn_food()
        self.done = False
        self.score = 0
        return self.get_state()

    def spawn_food(self):
        """Spawns food at a random location, ensuring it does not overlap the snake."""
        while True:
            food = (random.randint(0, WIDTH // GRID_SIZE - 1), random.randint(0, HEIGHT // GRID_SIZE - 1))
            if food not in self.snake:
                return food

    def step(self, action):
        """Takes an action and updates the game state."""
        if action == UP and self.direction != DOWN:
            self.direction = UP
        elif action == DOWN and self.direction != UP:
            self.direction = DOWN
        elif action == LEFT and self.direction != RIGHT:
            self.direction = LEFT
        elif action == RIGHT and self.direction != LEFT:
            self.direction = RIGHT

        # Move the snake
        head_x, head_y = self.snake[0]
        if self.direction == UP:
            head_y -= 1
        elif self.direction == DOWN:
            head_y += 1
        elif self.direction == LEFT:
            head_x -= 1
        elif self.direction == RIGHT:
            head_x += 1

        new_head = (head_x, head_y)

        # Check for collisions
        if (
            new_head in self.snake
            or head_x < 0
            or head_y < 0
            or head_x >= WIDTH // GRID_SIZE
            or head_y >= HEIGHT // GRID_SIZE
        ):
            self.done = True
            return self.get_state(), -10, self.done

        # Check if food is eaten
        if new_head == self.food:
            self.snake.insert(0, new_head)
            self.food = self.spawn_food()
            self.score += 1
            reward = 10
        else:
            self.snake.insert(0, new_head)
            self.snake.pop()
            reward = 0

        # Encourage moving closer to food
        reward += self.calculate_distance_reward(new_head)

        return self.get_state(), reward, self.done

    def calculate_distance_reward(self, new_head):
        """Gives a reward or penalty based on proximity to the food."""
        food_x, food_y = self.food
        head_x, head_y = new_head
        distance = abs(food_x - head_x) + abs(food_y - head_y)
        return -1 if distance > abs(food_x - self.snake[0][0]) + abs(food_y - self.snake[0][1]) else 1

    def get_state(self):
        """Returns a more detailed state representation."""
        head_x, head_y = self.snake[0]
        food_x, food_y = self.food

        # Check if the snake is near any boundaries or obstacles
        left_obstacle = (head_x <= 0 or (head_x - 1, head_y) in self.snake)
        right_obstacle = (head_x >= WIDTH // GRID_SIZE - 1 or (head_x + 1, head_y) in self.snake)
        top_obstacle = (head_y <= 0 or (head_x, head_y - 1) in self.snake)
        bottom_obstacle = (head_y >= HEIGHT // GRID_SIZE - 1 or (head_x, head_y + 1) in self.snake)

        # Add distance to food as part of the state representation
        distance_to_food = abs(food_x - head_x) + abs(food_y - head_y)

        return (
            food_x - head_x,  # Horizontal distance to food
            food_y - head_y,  # Vertical distance to food
            self.direction,   # Current direction of snake
            left_obstacle,    # Left boundary check
            right_obstacle,   # Right boundary check
            top_obstacle,     # Top boundary check
            bottom_obstacle,  # Bottom boundary check
            distance_to_food   # Distance to food
        )

    def render(self):
        """Draws the game state."""
        screen.fill(BLACK)

        # Draw snake
        for segment in self.snake:
            pygame.draw.rect(
                screen, GREEN, (segment[0] * GRID_SIZE, segment[1] * GRID_SIZE, GRID_SIZE, GRID_SIZE)
            )

        # Draw food
        pygame.draw.rect(
            screen, RED, (self.food[0] * GRID_SIZE, self.food[1] * GRID_SIZE, GRID_SIZE, GRID_SIZE)
        )

        pygame.display.flip()


def train_q_learning_with_visualization():
    game = SnakeGame()
    q_table = defaultdict(float)
    epsilon = EPSILON_START
    scores = []

    fig, ax = plt.subplots()
    plt.ion()
    ax.set_xlim(0, EPISODES)
    ax.set_ylim(0, 50)
    ax.set_xlabel("Episodes")
    ax.set_ylabel("Scores")
    ax.set_title("Real-Time Training Progress")
    line, = ax.plot([], [], label="Score")
    plt.legend()

    for episode in range(EPISODES):
        state = game.reset()
        total_reward = 0

        while not game.done:
            if random.random() < epsilon:
                action = random.randint(0, 3)
            else:
                action = max(range(4), key=lambda a: q_table[(state, a)])

            next_state, reward, done = game.step(action)
            total_reward += reward

            # Q-learning update
            q_value = q_table[(state, action)]
            max_next_q = max(q_table[(next_state, a)] for a in range(4))
            q_table[(state, action)] = q_value + ALPHA * (reward + GAMMA * max_next_q - q_value)

            state = next_state

            # Render the game
            game.render()
            clock.tick(FPS)

        scores.append(game.score)
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

        # Debug: Print the Q-value for specific state-action pair
        if episode % 100 == 0:
            print(f"Episode {episode + 1}/{EPISODES}, Score: {game.score}, Epsilon: {epsilon:.3f}")

        # Update graph
        line.set_xdata(range(len(scores)))
        line.set_ydata(scores)
        ax.relim()
        ax.autoscale_view()
        plt.pause(0.001)

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    train_q_learning_with_visualization()
