from collections import deque
import numpy as np

class DFS():
    def __init__(self, matrix, goal):
        self.Cost = 0
        self.Path = []
        self.depth = 0
        self.matrix = matrix
        self.goal = goal
        self.explored = set()
        self.expanded_nodes = set()
        self.parent = {}
        self.moves = []
        self.depth_tracker = {}
    
    def matrix_to_tuple(self, matrix):
        return tuple(map(tuple, matrix))
    
    def get_neighbors(self, Current_State):
        neighbors = []
        Row, Col = np.where(Current_State == 0)
        Row, Col = Row[0], Col[0]
        
        if Row != 0:
            new_State = Current_State.copy()
            new_State[Row][Col], new_State[Row-1][Col] = new_State[Row-1][Col], new_State[Row][Col]
            neighbors.append(new_State)
        
        if Col != 0:
            new_State = Current_State.copy()
            new_State[Row][Col], new_State[Row][Col-1] = new_State[Row][Col-1], new_State[Row][Col]
            neighbors.append(new_State)
        
        if Row != 2:
            new_State = Current_State.copy()
            new_State[Row][Col], new_State[Row+1][Col] = new_State[Row+1][Col], new_State[Row][Col]
            neighbors.append(new_State)
        
        if Col != 2:
            new_State = Current_State.copy()
            new_State[Row][Col], new_State[Row][Col+1] = new_State[Row][Col+1], new_State[Row][Col]
            neighbors.append(new_State)
        
        return neighbors
    
    def get_move_direction(self, state1, state2):
        """Determine which direction the blank moved"""
        pos1 = np.where(state1 == 0)
        pos2 = np.where(state2 == 0)
        row1, col1 = pos1[0][0], pos1[1][0]
        row2, col2 = pos2[0][0], pos2[1][0]
        
        if row2 < row1:
            return "Up"
        elif row2 > row1:
            return "Down"
        elif col2 < col1:
            return "Left"
        elif col2 > col1:
            return "Right"
        return ""
    
    def DFS_Algorithm(self):
        DFS_deque = deque([self.matrix])
        initial_tuple = self.matrix_to_tuple(self.matrix)
        self.parent[initial_tuple] = None
        self.depth_tracker[initial_tuple] = 0
        self.expanded_nodes.add(initial_tuple) 
        
        max_depth = 0
        
        while len(DFS_deque) > 0:
            Current_State = DFS_deque.pop()
            Current_State_as_tuple = self.matrix_to_tuple(Current_State)
            
            if Current_State_as_tuple in self.explored:
                continue
                
            self.explored.add(Current_State_as_tuple)
            
            current_depth = self.depth_tracker[Current_State_as_tuple]
            max_depth = max(max_depth, current_depth)
            
            if np.array_equal(Current_State, self.goal):
                self.create_path(Current_State_as_tuple)
                self.depth = max_depth
                self.Cost = len(self.Path) - 1
                return 1
            
            neighbors = self.get_neighbors(Current_State)
            
            for neighbor in neighbors:
                new_tuple = self.matrix_to_tuple(neighbor)
                
                if new_tuple not in self.explored:
                    if new_tuple not in self.parent:
                        self.parent[new_tuple] = Current_State_as_tuple
                        self.depth_tracker[new_tuple] = current_depth + 1
                        self.expanded_nodes.add(new_tuple) 
                        DFS_deque.append(neighbor)
        
        self.depth = max_depth
        return 0
    
    def create_path(self, tuple_value):
        path = []
        Current_State = tuple_value
        while Current_State is not None:
            path.append(np.array(Current_State))
            Current_State = self.parent[Current_State]
        
        self.Path = path[::-1]
        
        self.moves = []
        for i in range(len(self.Path) - 1):
            move = self.get_move_direction(self.Path[i], self.Path[i+1])
            self.moves.append(move)


