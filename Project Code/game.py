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


    NOTE:       Run the agent.py file to run the project
    
"""
###############################################################################


# Import statements
import pygame as pg
import numpy as np
import random
import sys
from enum import Enum
from collections import namedtuple


# Initialise the game
pg.init()
font = pg.font.Font('arial.ttf', 25)

# Define possible directions
class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4
    
Point = namedtuple('Point', 'x, y')

# Define RGB colours and game operations
# Adjust SPEED to increase/decrease game speed
WHITE   = (255, 255, 255)
RED     = (200,0,0)
BLUE1   = (0, 0, 255)
BLUE2   = (0, 100, 255)
BLACK   = (40,40,40)

BLOCK_SIZE = 20
SPEED = 300

class SnakeGameAI:
    
    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        self.display = pg.display.set_mode((self.w, self.h))
        pg.display.set_caption('Snake')
        self.clock = pg.time.Clock()
        self.reset()
        
    def reset(self):
        # Reset to initial game state when done
        self.direction = Direction.RIGHT
        
        self.head   = Point(self.w/2, self.h/2)
        self.snake  = [self.head, 
                       Point(self.head.x-BLOCK_SIZE, self.head.y),
                       Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]
        
        self.score  = 0
        self.food   = None
        self._place_food()
        self.frame_iteration = 0
        
    def _place_food(self):
        # Place the food bit randomly in the window
        x   = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE 
        y   = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        self.food = Point(x, y)
        
        if self.food in self.snake:
            self._place_food()
        
    def play_step(self, action):
        # Add an iteration count
        self.frame_iteration += 1
        
        # Allow the user to quit the session whenever desired
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                sys.exit()
      
        # Conduct move with snake
        self._move(action) # update the head
        self.snake.insert(0, self.head)
        
        # See if the game is over
        reward      = 0
        game_over   = False
        
        if self.is_collision() or self.frame_iteration > 100*len(self.snake):
            game_over   = True
            reward      = -10
            return reward, game_over, self.score
            
        # Elsewise place new food or conduct new move
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()
        
        # Update ui and clock
        self._update_ui()
        self.clock.tick(SPEED)
        
        # Finally return game over and score
        return reward, game_over, self.score
    
    def is_collision(self, pt=None):
        # Check if a collision happened
        if pt is None:
            pt = self.head
            
        # hits boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True

        # hits itself
        if pt in self.snake[1:]:
            return True
        
        return False
        
    def _update_ui(self):
        # Update the UI/Game window by drawing everything to window
        self.display.fill(BLACK)
        
        for pt in self.snake:
            pg.draw.rect(self.display, BLUE1, pg.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pg.draw.rect(self.display, BLUE2, pg.Rect(pt.x+4, pt.y+4, 12, 12))
            
        pg.draw.rect(self.display, RED, pg.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        pg.display.flip()
        
    def _move(self, action):
        # Conduct move in either: [straight, right, left]
        
        # Cycle clockwise through possible directions
        clock_wise  = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx         = clock_wise.index(self.direction)
        
        # Move straight ahead
        if np.array_equal(action, [1,0,0]):
            new_dir = clock_wise[idx]
        
        # Elsewise make a right turn
        elif np.array_equal(action, [0,1,0]):
            next_idx    = (idx + 1) % 4
            new_dir     = clock_wise[next_idx]
            
        # Elsewise make a left turn
        else:
            next_idx    = (idx - 1) % 4
            new_dir     = clock_wise[next_idx] 
        
        # Update the direction
        self.direction  = new_dir
        
        # Update coordinates and blocks
        x = self.head.x
        y = self.head.y
        
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE
            
        self.head = Point(x, y)