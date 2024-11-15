import numpy as np
from gymnasium import Env
from pathlib import Path
import os
import pygame # pygame is used for rendering

STARTING = 2
FINISHING = 3

# Race track environment
class RaceTrack(Env):

    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 20}



    def __init__(self, track_map:str, render_mode:str=None, size:int=2):
        self.size = size # the size of cells
        
        assert track_map in ['a', 'b']
        assert render_mode is None or render_mode in self.metadata['render_modes']
        self.render_mode = render_mode

        # reading a track map
        
        filename = 'track_a.npy' if track_map == 'a' else 'track_b.npy'
        filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)

        with open( filepath, 'rb') as f:
            self.track_map = np.load(f)
            print("Carte chargée")

        # Initialize parameters for pygame
        self.window_size = self.track_map.shape
        # Pygame's coordinate if the transpose of that of numpy
        self.window_size = (self.window_size[1] * self.size, self.window_size[0] * self.size)
        self.window = None # window for pygame rendering
        self.clock = None # clock for pygame ticks
        self.truncated = False


        # Get start states
        self.start_states = np.dstack(np.where(self.track_map==STARTING))[0]
        print(f"Positions de départ trouvées : {len(self.start_states)}")  # Vérifiez les états de départ

        self.nS = (*self.track_map.shape, 5, 9) # observation space
        self.nA = 9 # action space
        self.state = None # Initialize state
        self.speed = None # Initialize speed

        # Mapping the integer action to acceleration tuple
        self._action_to_acceleration = {
            0: (-1, -1),
            1: (-1, 0),
            2: (-1, 1),
            3: (0, -1),
            4: (0, 0),
            5: (0, 1),
            6: (1, -1),
            7: (1, 0),
            8: (1, 1)
        }

    # Get observation
    def _get_obs(self):
        return (*self.state, *self.speed)

    # Get info, always return None in our case
    def _get_info(self):
        return None

    # Check if the race car go accross the finishing line
    def _check_finish(self):
        finish_states = np.where(self.track_map == FINISHING)
        rows = finish_states[0]
        col = finish_states[1][0]
        if self.state[0] in rows and self.state[1] >= col:
            return True
        return False

    # Check if the track run out of the track
    def _check_out_track(self, next_state):
        row, col = next_state
        H, W = self.track_map.shape
        # If the car go out of the boundary 
        if row < 0 or row >= H or col < 0 or col >= W:
            return True
        # Check if the car run into the gravels
        if self.track_map[next_state[0], next_state[1]] == 0:
            return True
        
        # Check if part of the path run into gravels
        for row_step in range(self.state[0], row, -1):
            if self.track_map[row_step, self.state[1]] == 0: return True
        for col_step in range(self.state[1], col, 1 if col > self.state[1] else -1):
            if self.track_map[row, col_step] == 0: return True

        return False


    # reset the car to one of the starting positions
    def reset(self):
        # Select start position randomly from the starting line
        start_idx = np.random.choice(self.start_states.shape[0])
        self.state = self.start_states[start_idx]
        self.speed = (0, 0)

        if self.render_mode == 'human':
            self.render(self.render_mode)
        return self._get_obs(), self._get_info()


    # take actions
    def step(self, action):
        # Get new acceleration and updated position
        new_state = np.copy(self.state)
        y_act, x_act = self._action_to_acceleration[action]
        
        temp_y_acc = self.speed[0] + y_act
        temp_x_acc = self.speed[1] + x_act
        
        if temp_y_acc < -4: temp_y_acc = -4
        if temp_y_acc > 0: temp_y_acc = 0 # Avoid the car from going backward
        if temp_x_acc < -4: temp_x_acc = -4
        if temp_x_acc > 4: temp_x_acc = 4
        
        new_state[0] += temp_y_acc
        new_state[1] += temp_x_acc
        
        terminated = False
        # check if next position crosses the finishing line
        if self._check_finish():
            terminated = True 
        # check if next postion locates in invalid places
        elif self._check_out_track(new_state):
            self.reset()
        else:
            self.state = new_state
            self.speed = (temp_y_acc, temp_x_acc)

        if self.render_mode == 'human':
            self.render(self.render_mode)
        
        return self._get_obs(), -1, terminated, self.truncated


    # visualize race map
    def render(self, mode):

        if self.window is None:
            print("Initialisation de Pygame...")
            pygame.init()
            print("Pygame initialisé.")
            pygame.init()
            pygame.display.set_caption('Race Track')
            if mode == 'human':
                self.window = pygame.display.set_mode(self.window_size)

        if self.clock is None:
            self.clock = pygame.time.Clock()

        rows, cols = self.track_map.shape
        self.window.fill((255, 255, 255))


        # Draw the map
        for row in range(rows):
            for col in range(cols):
                cell_val = self.track_map[row, col]
                # Draw finishing cells
                if cell_val == FINISHING:
                    fill = (235, 52, 52)
                    pygame.draw.rect(self.window, fill, (col * self.size, row * self.size, self.size, self.size), 0)
                # Draw starting cells
                elif cell_val == STARTING:
                    fill = (61, 227, 144)            
                    pygame.draw.rect(self.window, fill, (col * self.size, row * self.size, self.size, self.size), 0)

                color =(250, 250, 250)
                # Draw gravels
                if cell_val == 0.0:
                    color = (0, 0, 0)

                if cell_val > 10:
                    # Normalize cell_val to a range between 0 and 1
                    normalized_val = (cell_val - 60) / (85- 60)
                    # Apply a stretching factor (e.g., 2x or more) to increase the contrast
                    # Map to a blue-to-red gradient
                    # Example: low elevation (blue), high elevation (red)
                    blue = int(255 * (1 - normalized_val))  # More blue at lower elevations
                    red = int(255 * normalized_val)         # More red at higher elevations
                    color = (red, 0, blue)
                    fill = (red, 0, blue)
                
                    pygame.draw.rect(self.window, color, (col * self.size, row * self.size, self.size, self.size), 0)
        
        # Draw the car
        pygame.draw.rect(self.window, (86, 61, 227), (self.state[1] * self.size, self.state[0] * self.size, self.size, self.size), 0)

        if mode == "human":
            pygame.display.update()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.window = None
                    pygame.quit()
                    self.truncated = True
            self.clock.tick(self.metadata['render_fps'])