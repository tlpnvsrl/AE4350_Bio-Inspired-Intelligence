###############################################################################
"""
    Course:     Bio-Inspired Intelligence and Learning for Aerospace- 
                Applications

    Code:       AE4350 
    Year:       2023/2024 Q5
    Topic:      Applying Deep Q-Learning to the Snake Game
 
    Student:    Lars van Pelt
    Stud. no:   5629632
    Email:      L.H.vanpelt@student.tudelft.nl    


    NOTE:       Run this file to run the project
    
"""
###############################################################################

# Import statements
import numpy as np
import pygame as pg
import matplotlib.pyplot as plt
import torch
import random
import time
import os
import sys

# Delete this to make plots visible
import matplotlib
matplotlib.use('Agg')

from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer


# Agent and neural network parameters
MAX_MEMORY      = 100_000
BATCH_SIZE      = 1000
HIDDEN_NEURONS  = 256
alpha           = 0.001     
no_episodes     = 2000

epsilon_decay   = 0.95
gamma           = 0.9

class Agent:
    
    def __init__(self):
        self.no_games       = 0
        self.epsilon_max    = 1.0
        self.epsilon_min    = 0.01
        self.epsilon_decay  = epsilon_decay
        self.epsilon    = self.epsilon_max
        self.gamma      = gamma 
        self.memory     = deque(maxlen=MAX_MEMORY) # Pops element on left when memory gets exceeded
        self.model      = Linear_QNet(11, HIDDEN_NEURONS, 3)
        self.trainer    = QTrainer(self.model, alpha, self.gamma)
        
    def get_state(self, game):
        # Get snake head
        head = game.snake[0]
        
        # Check possible directions besides snake for boundaries
        # Note: 20 is BLOCK SIZE
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        # Define current direction
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN
        
        # Define list with all 11 possible states
        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),
        
            # Danger right
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),
        
            # Danger left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
            ]
        
        # Return list as np array and convert boolean logic to ints
        return np.array(state, dtype=int)
    
    def decay_epsilon(self, episode):
        # Apply epsilon decay rate for greedy strategy
        self.epsilon = max(self.epsilon_min, self.epsilon_max * (self.epsilon_decay ** episode))
    
    def get_action(self, state, episode):
        self.decay_epsilon(episode)
        final_move  = [0,0,0]
        if random.random() < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
            
        return final_move
            
    
    def remember(self, state, action, reward, next_state, gameover):
        # Append data in deque to replay memory
        self.memory.append((state, action, reward, next_state, gameover)) 
    
    def train_long_memory(self):
        # Take variables from batch from memory
        if len(self.memory) > BATCH_SIZE:
            # Get list of tuples
            mini_sample = random.sample(self.memory, BATCH_SIZE) 
        else:
            mini_sample = self.memory
            
        states, actions, rewards, next_states, gameovers = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, gameovers)
            
    def train_short_memory(self, state, action, reward, next_state, gameover):
        # Perform only one step
        self.trainer.train_step(state, action, reward, next_state, gameover)
        
    def save_scores(self, directory, plot_scores, plot_mean_scores):
        # Save the scores to numpy arrays
        np.save(os.path.join(directory, 'plot_scores.npy'), plot_scores)
        np.save(os.path.join(directory, 'plot_mean_scores.npy'), plot_mean_scores)
        
    def make_train_dir(self, BATCH_SIZE, HIDDEN_NEURONS, alpha, gamma, epsilon_decay):
        # Generate name for directory
        dir_name = f"bsize_{BATCH_SIZE}_neurons_{HIDDEN_NEURONS}_alpha_{alpha}_gamma_{gamma}_epsilon_{epsilon_decay}_episodes_{no_episodes}"
        
        # Create directory if it doesn't exist
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        
        train_dir = os.path.join(dir_name)
        
        # Create text file to log parameters
        params_file = os.path.join(train_dir, "params.txt")
        with open(params_file, "w") as f:
            f.write(f"BATCH SIZE: {BATCH_SIZE}\n")
            f.write(f"HIDDEN LAYER SIZE: {HIDDEN_NEURONS}\n")
            f.write(f"ALPHA: {alpha}\n")
            f.write(f"GAMMA: {gamma}\n")
            f.write(f"EPSILON_DECAY: {epsilon_decay}\n")
        
        return train_dir
        
def train():
    plot_scores         = []
    plot_mean_scores    = []
    total_score         = 0
    record              = 0
    agent               = Agent()
    game                = SnakeGameAI()
    
    start_time  = time.time()
    train_dir   = agent.make_train_dir(BATCH_SIZE, HIDDEN_NEURONS, alpha, agent.gamma, agent.epsilon_decay)
    
    while agent.no_games < no_episodes:
        # Get old state
        state_old = agent.get_state(game)
        
        # get move
        final_move = agent.get_action(state_old, agent.no_games)

        # Perform move and get new state
        reward, gameover, score = game.play_step(final_move)
        state_new = agent.get_state(game)
        
        # Train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, gameover)
        
        # Remember, so store in memory/deque
        agent.remember(state_old, final_move, reward, state_new, gameover)
        
        if gameover:
            # Train long memory, i.e. replay memory
            game.reset()
            agent.no_games += 1
            agent.train_long_memory()
            
            if score > record:
                # A new high score has been achieved
                record = score
                agent.model.save(train_dir)
                
            print('Game', agent.no_games, 'Epsilon: %.4f' % agent.epsilon, 'Score', score, "Record:", record)
            
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.no_games
            plot_mean_scores.append(mean_score)
                    
    agent.save_scores(train_dir, plot_scores, plot_mean_scores)
    pg.quit()
    
    # Present time training session took
    end_time    = time.time()
    duration    = end_time - start_time
    if duration > 60:
        minutes = int(duration // 60)
        seconds = int(duration % 60)
        print(f"Training completed in: {minutes} minutes and {seconds} seconds")
    else:
        print(f"Training completed in: {duration:.2f} seconds")
    
    plt.figure(1)
    plt.plot(list(range(1, no_episodes + 1)), plot_scores)
    plt.xlim((1, no_episodes))
    plt.xlabel("Episodes")
    plt.ylabel("Score")
    plt.title("Scores per Episode")
    plt.grid(True, alpha = 0.5)
    plot_filename = os.path.join(train_dir, 'scores_plot.png')
    plt.savefig(plot_filename, dpi=300)
    plt.close()
    
    plt.figure(2)
    plt.plot(list(range(1, no_episodes + 1)), plot_mean_scores)
    plt.xlim((1, no_episodes))
    plt.xlabel("Episodes")
    plt.ylabel("Mean Score")
    plt.title("Mean Scores per Episode")
    plt.grid(True, alpha = 0.5)
    plot_filename = os.path.join(train_dir, 'mean_scores_plot.png')
    plt.savefig(plot_filename, dpi=300)
    plt.close()
    
if __name__ == '__main__':
    train()
