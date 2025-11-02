from collections import deque
from Tree import Tree 
import numpy as np

class BFS():
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
    
    def find_zero(self, current):
        for i in range(3):
            for j in range(3):
                if current[i][j] == 0:
                    return (i, j)
    
    def valid(self, x, y):
        return 0 <= x < 3 and 0 <= y < 3
    
    def get_move_name(self, dx, dy):
        if dx == -1 and dy == 0:
            return "Up"
        elif dx == 1 and dy == 0:
            return "Down"
        elif dx == 0 and dy == -1:
            return "Left"
        elif dx == 0 and dy == 1:
            return "Right"
        return None
    
    def reconstruct_path(self, parent, move_parent, goal_state):
        path = []
        moves = []
        key = tuple(goal_state.flatten())    
        while key is not None:
            state = np.array(key).reshape(3, 3)
            path.append(state)
            if key in move_parent:
                moves.append(move_parent[key])
            key = parent[key]    
        path.reverse()
        moves.reverse()
        return path, moves
    
    def BFS_Algorithm(self):
        start = np.copy(self.matrix)
        q = deque()
        q.append((start, 0))  
        visited = set()
        visited.add(tuple(start.flatten()))  
        parent = {}
        parent[tuple(start.flatten())] = None
        move_parent = {} 
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        while q:
            current, current_depth = q.popleft()
            
            if np.array_equal(current, self.goal):
                self.path, self.path_to_goal = self.reconstruct_path(parent, move_parent, self.goal)
                self.cost = len(self.path) - 1  
                self.depth = current_depth
                return self.path, self.cost, self.nodes_expanded, self.depth
            
            x, y = self.find_zero(current)    
            
            for dx, dy in directions:
                new_x = x + dx 
                new_y = y + dy
                
                if self.valid(new_x, new_y):
                    child = np.copy(current)
                    child[x, y], child[new_x, new_y] = child[new_x, new_y], child[x, y]
                    key = tuple(child.flatten())
                    
                    if key not in visited:
                        visited.add(key)
                        self.explored.add(key)
                        q.append((child, current_depth + 1))  
                        parent[key] = tuple(current.flatten())
                        move_parent[key] = self.get_move_name(dx, dy)  
                        self.nodes_expanded+=1
        
        return [], 0, self.nodes_expanded, 0



