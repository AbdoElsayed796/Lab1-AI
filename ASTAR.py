import heapq
import numpy as np
from collections import deque
import math
import itertools

class AStar():
    def __init__(self, matrix, goal):
        self.cost = 0
        self.path = []
        self.path_to_goal = []  
        self.nodes_expanded = 0
        self.depth = 0
        self.matrix = matrix
        self.goal = goal
        self.explored = set()
        self.parent = {}
        self.counter = itertools.count()  
    
    def find_zero(self, current):
        for i in range(3):
            for j in range(3):
                if current[i][j] == 0:
                    return (i, j)
    
    def valid(self, x, y):
        return 0 <= x < 3 and 0 <= y < 3
    
    def get_move_direction(self, zero_before, zero_after):
        dx = zero_after[0] - zero_before[0]
        dy = zero_after[1] - zero_before[1]
        
        if dx == -1: return "Up"
        elif dx == 1: return "Down"
        elif dy == -1: return "Left"
        elif dy == 1: return "Right"
        return "Unknown"
    
    def reconstruct_path(self, parent, goal_state):
        path_states = []
        path_directions = []
        
        key = tuple(goal_state.flatten())
        while key is not None:
            state = np.array(key).reshape(3, 3)
            path_states.append(state)
            key = parent[key]
        path_states.reverse()
        
    
        for i in range(1, len(path_states)):
            zero_before = self.find_zero(path_states[i-1])
            zero_after = self.find_zero(path_states[i])
            direction = self.get_move_direction(zero_before, zero_after)
            path_directions.append(direction)
        
        return path_states, path_directions
    
    def AStar_Algorithm(self, heuristic_function):
        start = np.copy(self.matrix)
        
        # Priority queue: (f_cost, counter, state_key, g_cost)
        # Using state_key (tuple) instead of state (numpy array) to avoid comparison issues
        open_set = []
        initial_g = 0
        initial_f = initial_g + heuristic_function(start, self.goal)
        start_key = tuple(start.flatten())
        heapq.heappush(open_set, (initial_f, next(self.counter), start_key, initial_g))
        
        g_costs = {}  # g(n) costs (always increases by 1 per move)
        parent = {}
        in_open_set = set()  # Track what's currently in open set
        
        g_costs[start_key] = initial_g
        parent[start_key] = None
        in_open_set.add(start_key)
        
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        while open_set:
            current_f, _, current_key, current_g = heapq.heappop(open_set)
            in_open_set.discard(current_key)
            
            if current_g > g_costs.get(current_key, float('inf')):
                continue
            
            current = np.array(current_key).reshape(3, 3)
            
            self.explored.add(current_key)
            self.nodes_expanded += 1
            
            if np.array_equal(current, self.goal):
                self.path, self.path_to_goal = self.reconstruct_path(parent, self.goal)
                self.cost = current_g 
                self.depth = current_g
                return self.path, self.cost, self.nodes_expanded, self.depth, self.path_to_goal
            
            x, y = self.find_zero(current)
            
            for dx, dy in directions:
                new_x = x + dx
                new_y = y + dy
                
                if self.valid(new_x, new_y):
                    child = np.copy(current)
                    child[x, y], child[new_x, new_y] = child[new_x, new_y], child[x, y]
                    child_key = tuple(child.flatten())
        
                    if child_key in self.explored:
                        continue

                    new_g = current_g + 1

                    if child_key not in g_costs or new_g < g_costs[child_key]:
                        g_costs[child_key] = new_g
                        parent[child_key] = current_key

                        h_cost = heuristic_function(child, self.goal)
                        f_cost = new_g + h_cost
                        
                        heapq.heappush(open_set, (f_cost, next(self.counter), child_key, new_g))
                        in_open_set.add(child_key)
        
        return [], 0, self.nodes_expanded, 0, []


def manhattan_distance(state, goal):
    distance = 0
    for i in range(3):
        for j in range(3):
            if state[i][j] != 0: 
                value = state[i][j]    
                goal_positions = np.where(goal == value)
                goal_i, goal_j = goal_positions[0][0], goal_positions[1][0]
                distance += abs(i - goal_i) + abs(j - goal_j)
    return distance


def euclidean_distance(state, goal):
    distance = 0
    for i in range(3):
        for j in range(3):
            if state[i][j] != 0:
                value = state[i][j]
                goal_positions = np.where(goal == value)
                goal_i, goal_j = goal_positions[0][0], goal_positions[1][0]
                distance += math.sqrt((i - goal_i)**2 + (j - goal_j)**2)
    return distance
