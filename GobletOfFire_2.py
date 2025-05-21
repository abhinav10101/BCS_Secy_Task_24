import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame
from gymnasium.envs.registration import register
from collections import deque

import time

# Reading the txt file 
with open('V1.txt', 'r') as f:
    grid = [list(line.strip()) for line in f]

# Precompute blocked cells
blocked_cells = {(i, j) for i, row in enumerate(grid) for j, cell in enumerate(row) if cell == 'X'}

# Environment constants
HEIGHT = 15
WIDTH = 20
MAX_STEPS = 1000

class GridWorldEnv(gym.Env):
    def __init__(self, height=HEIGHT, width=WIDTH):
        self.height = height
        self.width = width
        self.action_space = spaces.Discrete(4)
        self.max_steps = MAX_STEPS
        self.current_step = 0
        self.window_size = 512
        self.window = None
        self.clock = None
        

        self.action_to_dir = [
            np.array([1, 0]),   # move up
            np.array([0, 1]),    # move right
            np.array([-1, 0]),   # move down
            np.array([0, -1])    # move left
        ]
        
        self.observation_space = spaces.Dict({
            "agent": spaces.Box(low=0, high=np.array([self.height-1, self.width-1]), dtype=int),
            "target": spaces.Box(low=0, high=np.array([self.height-1, self.width-1]), dtype=int),
            "DeathEater1": spaces.Box(low=0, high=np.array([self.height-1, self.width-1]), dtype=int),
            "DeathEater2": spaces.Box(low=0, high=np.array([self.height-1, self.width-1]), dtype=int)
        })

        self.valid_positions = [(i, j) for i in range(height) for j in range(width) 
                              if (i, j) not in blocked_cells]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        positions = np.random.choice(len(self.valid_positions), 4, replace=False)
        self.agentlocation = np.array(self.valid_positions[positions[0]])
        self.targetlocation = np.array(self.valid_positions[positions[1]])
        self.deathlocation_first = np.array(self.valid_positions[positions[2]])
        self.deathlocation_second = np.array(self.valid_positions[positions[3]])

        self.current_step = 0
        return self.getobs(), {}

    def bfs(self):
        visited = set()
        queue = deque()
        start = tuple(self.deathlocation_first)
        queue.append((start[0], start[1], None))
        visited.add(start)

        while queue:
            x, y, first_step = queue.popleft()
    
            if (x, y) == tuple(self.agentlocation):
                return first_step
                
            
            for action, (dx, dy) in enumerate(self.action_to_dir):
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.height and 0 <= ny < self.width:
                    new_pos = (nx, ny)
                    if new_pos not in blocked_cells and new_pos not in visited:
                        visited.add(new_pos)
                        new_first = action if (x, y) == start else first_step
                        queue.append((nx, ny, new_first))
        return None

    def getobs(self):
        return {
            "agent": self.agentlocation,
            "target": self.targetlocation,
            "DeathEater1": self.deathlocation_first,
            "DeathEater2" : self.deathlocation_second
        }

    def find_reward(self, terminated, eaten, old_agent_pos, old_death_pos_first,old_death_pos_second):
        max_distance = self.height + self.width - 2
        
        # Distance to target
        new_dist_target = np.sum(np.abs(self.agentlocation - self.targetlocation))
        old_dist_target = np.sum(np.abs(old_agent_pos - self.targetlocation))
        target_progress = (old_dist_target - new_dist_target) / max_distance
        
        # Distance to Death Eater 1
        new_dist_death_first = np.sum(np.abs(self.agentlocation - self.deathlocation_first))
        old_dist_death_first = np.sum(np.abs(old_agent_pos - old_death_pos_first))
        escapeprogress_first = (old_dist_death_first - new_dist_death_first) / max_distance

        # Distance to Death Eater 2
        new_dist_death_second = np.sum(np.abs(self.agentlocation - self.deathlocation_second))
        old_dist_death_second = np.sum(np.abs(old_agent_pos - old_death_pos_second))
        escapeprogress_second = (old_dist_death_second - new_dist_death_second) / max_distance

        
        reward = 0
        reward += 100 * target_progress  
        reward += 2 * (1 - new_dist_target/max_distance)  
        reward -= 30 * (1 - new_dist_death_first/max_distance)  
        reward -= 30*(1-new_dist_death_second/max_distance)
        reward -= 20 * escapeprogress_first  
        reward -= 20 * escapeprogress_second
        
        if terminated:
            reward += 300  
        if eaten:
            reward -= 200  
            
        # Time penalty
        reward -= 0.5 * (self.current_step / self.max_steps)
        
        return reward

    def step(self, action_agent):
        self.current_step += 1
        truncated = self.current_step >= self.max_steps
        
        
        old_agent_pos = self.agentlocation
        old_death_pos_first = self.deathlocation_first
        old_death_pos_second = self.deathlocation_second

        #Changing first death eaters location
        death_action_first = self.bfs()
        if death_action_first is None:
            death_action_first = np.random.randint(4)
        death_move_first = self.action_to_dir[death_action_first]
        new_death_pos_first = np.clip(self.deathlocation_first + death_move_first, [0, 0], [self.height-1, self.width-1])
        if tuple(new_death_pos_first) not in blocked_cells:
            self.deathlocation_first = new_death_pos_first

        #changing second death eaters location
        death_action_second = self.bfs()
        if death_action_second is None:
            death_action_second = np.random.randint(4)
        death_move_second = self.action_to_dir[death_action_second]
        new_death_pos_second = np.clip(self.deathlocation_second + death_move_second, [0, 0], [self.height-1, self.width-1])
        if tuple(new_death_pos_second) not in blocked_cells:
            self.deathlocation_second = new_death_pos_second

        # Agent movement
        agent_move = self.action_to_dir[action_agent]
        new_agent_pos = np.clip(self.agentlocation + agent_move, [0, 0], [self.height-1, self.width-1])
        
        if tuple(new_agent_pos) not in blocked_cells and not np.array_equal(new_agent_pos,old_death_pos_first) and not np.array_equal(new_agent_pos,old_death_pos_second):
            self.agentlocation = new_agent_pos
        else:
            # Find best alternative move if original is blocked
            best_pos = None
            max_distance = -1
            for action in range(4):
                move = self.action_to_dir[action]
                new_agent_location = np.clip(self.agentlocation + move, [0, 0], [self.height-1, self.width-1])
                if tuple(new_agent_location) not in blocked_cells and not np.array_equal(new_agent_location,old_death_pos_first) and not np.array_equal(new_agent_location,old_death_pos_second):
                    dist = max(np.sum(np.abs(new_agent_location - self.deathlocation_first)),np.sum(np.abs(new_agent_location - self.deathlocation_second)))
                    if dist > max_distance:
                        max_distance = dist
                        best_pos = new_agent_location
            if best_pos is not None:
                self.agentlocation = best_pos

        terminated = np.array_equal(self.agentlocation, self.targetlocation)
        eaten = np.array_equal(self.agentlocation, self.deathlocation_first) or np.array_equal(self.agentlocation, self.deathlocation_second)

        # Calculate reward
        reward = self.find_reward(terminated, eaten, old_agent_pos, old_death_pos_first,old_death_pos_second)
        
        info = {
            "Win": terminated,
            "Loss": eaten,
            "action": action_agent
        }
        
        return self.getobs(), reward, terminated, truncated, info

    # Rendering methods remain the same...
    def _render_frame(self):
        pygame.init()
        pygame.display.init()

        if self.window is None:
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None:
            self.clock = pygame.time.Clock()
        canvas = pygame.Surface((self.window_size,self.window_size))

        #Drawing the target
        canvas.fill((255,255,255))
        pix_rectangle_size = (self.window_size/self.height,self.window_size/self.width)
        pygame.draw.rect(canvas,(0,255,0),pygame.Rect(self.targetlocation[1]*pix_rectangle_size[1],self.targetlocation[0]*pix_rectangle_size[0],pix_rectangle_size[1],pix_rectangle_size[0]))
        
        #making the walls
        for x,y in blocked_cells:
            pygame.draw.rect(canvas,(255,0,0),pygame.Rect(y*pix_rectangle_size[1],x*pix_rectangle_size[0],pix_rectangle_size[1],pix_rectangle_size[0]))
        #Drawing circle for the agent
        center_y = int((self.agentlocation[0]+0.5)*pix_rectangle_size[0])
        center_x = int((self.agentlocation[1]+0.5)*pix_rectangle_size[1])
        radius = int(min(pix_rectangle_size[0],pix_rectangle_size[1])/2)
        pygame.draw.circle(canvas,(255,255,0),(center_x,center_y),radius)

        #Drawing circle for the death eater 1
        center_y = int((self.deathlocation_first[0]+0.5)*pix_rectangle_size[0])
        center_x = int((self.deathlocation_first[1]+0.5)*pix_rectangle_size[1])
        radius = int(min(pix_rectangle_size[0],pix_rectangle_size[1])/2)
        pygame.draw.circle(canvas,(128,0,128),(center_x,center_y),radius)

        #Drawing circle for the death eater 2
        center_y = int((self.deathlocation_second[0]+0.5)*pix_rectangle_size[0])
        center_x = int((self.deathlocation_second[1]+0.5)*pix_rectangle_size[1])
        radius = int(min(pix_rectangle_size[0],pix_rectangle_size[1])/2)
        pygame.draw.circle(canvas,(128,0,128),(center_x,center_y),radius)

        #drawing the grid
        for x in range(self.height + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_rectangle_size[0] * x),
                (self.window_size, pix_rectangle_size[0] * x),
                width=3,
            )
        for x in range(self.width + 1):
            pygame.draw.line(
                canvas,
                0,
                (pix_rectangle_size[1]* x, 0),
                (pix_rectangle_size[1] * x, self.window_size),
                width=3,
            )
        self.window.blit(canvas, (0, 0))
        pygame.display.update()
        pygame.event.pump()
        
    def render(self):
        self._render_frame()
        pygame.display.update()  # Update the display
        pygame.event.pump() 

# Register the environment
register(
    id='GridWorld-v0',
    entry_point=__name__ + ':GridWorldEnv'
)

# Q-learning functions
class QLearningAgent:
    def __init__(self, height, width):
        self.height = height
        self.width = width
        self.Q = np.zeros((height*width, height*width,height*width, height*width, 4))
        
    def get_state(self, pos):
        return pos[0] * self.width + pos[1]
        
    def update(self, state, death_state_first,death_state_second, target_state, action, reward, 
               next_state, next_death_state_first,next_death_state_second, next_target_state, 
               alpha, gamma):
        current_q = self.Q[target_state, death_state_first,death_state_second, state, action]
        max_next_q = np.max(self.Q[next_target_state, next_death_state_first,next_death_state_second, next_state])
        new_q = current_q + alpha * (reward + gamma * max_next_q - current_q)
        self.Q[target_state, death_state_first,death_state_second,state, action] = new_q
        
    def get_action(self, state, death_state_first,death_state_second, target_state, epsilon):
        if np.random.random() < epsilon:
            return np.random.randint(4)
        return np.argmax(self.Q[target_state, death_state_first,death_state_second, state])

def train(epsilon_start, alpha, gamma, episodes):
    agent = QLearningAgent(HEIGHT, WIDTH)
    env = gym.make('GridWorld-v0')
    
    epsilon = epsilon_start
    epsilon_decay = epsilon_start / (episodes * 0.8) 
    
    for episode in range(episodes):
        observation, info = env.reset()
        terminated = False
        truncated = False
        
        target_state = agent.get_state(observation["target"])
        
        while not terminated and not truncated:
            state = agent.get_state(observation["agent"])
            death_state_first = agent.get_state(observation["DeathEater1"])
            death_state_second = agent.get_state(observation["DeathEater2"])
            
            action = agent.get_action(state, death_state_first,death_state_second, target_state, epsilon)
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            next_state = agent.get_state(next_obs["agent"])
            next_death_state_first = agent.get_state(next_obs["DeathEater1"])
            next_death_state_second = agent.get_state(next_obs["DeathEater2"])
            next_target_state = agent.get_state(next_obs["target"])
            
            agent.update(state, death_state_first,death_state_second, target_state, action, reward,
                         next_state, next_death_state_first,next_death_state_second,next_target_state,
                         alpha, gamma)
            
            observation = next_obs
        
        if epsilon > 0.1:
            epsilon -= epsilon_decay
    
    env.close()
    return agent

def test(agent, episodes, epsilon):
    env = gym.make('GridWorld-v0')
    wins = 0
    
    for i in range(episodes):
        observation, info = env.reset()
        terminated = False
        truncated = False
        target_state = agent.get_state(observation["target"])
        
        while not terminated and not truncated:
            # env.render()
            # time.sleep(0.1)
            state = agent.get_state(observation["agent"])
            death_state_first = agent.get_state(observation["DeathEater1"])
            death_state_second = agent.get_state(observation["DeathEater2"])
            action = agent.get_action(state, death_state_first,death_state_second, target_state, epsilon)
            observation, reward, terminated, truncated, info = env.step(action)
            
            if info["Win"]:
                wins += 1
                break
                
    env.close()
    print(wins/episodes)

agent = train(0.9,0.2,0.96,3000)
test(agent,100,0.1)
