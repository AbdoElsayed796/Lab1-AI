from collections import deque
import numpy as np

class IDS():
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
    
    def IDS_search(self):
        total_expanded = set()  # Track all expanded nodes across iterations
        
        for depth_limit in range(1, 40):
            # Reset state for this depth iteration
            self.parent = {}
            self.depth_tracker = {}
            self.moves = []
            self.Path = []
            
            # Run DLS with current depth limit
            isSolution = self.DLS_Algorithm(max_depth=depth_limit)
            
            # Accumulate expanded nodes
            total_expanded.update(self.expanded_nodes)
            
            if isSolution:
                self.expanded_nodes = total_expanded  # Set total count
                return True
        
        self.expanded_nodes = total_expanded
        return False

    def DLS_Algorithm(self, max_depth=None):
        Ids_deque = deque([(self.matrix, 0, None)])  # Store (state, depth, parent) tuples
        initial_tuple = self.matrix_to_tuple(self.matrix)
        self.expanded_nodes = set([initial_tuple])  # Reset for this iteration
        
        current_max_depth = 0
        visited_at_depth = {}  # Track the depth at which we visited each state
        
        while len(Ids_deque) > 0:
            Current_State, current_depth, parent_tuple = Ids_deque.pop()
            Current_State_as_tuple = self.matrix_to_tuple(Current_State)
            
            # Skip if we've already visited this state at this depth or shallower
            if Current_State_as_tuple in visited_at_depth and visited_at_depth[Current_State_as_tuple] <= current_depth:
                continue
            
            visited_at_depth[Current_State_as_tuple] = current_depth
            
            # Update parent
            if Current_State_as_tuple not in self.parent or current_depth < self.depth_tracker.get(Current_State_as_tuple, float('inf')):
                self.parent[Current_State_as_tuple] = parent_tuple
                self.depth_tracker[Current_State_as_tuple] = current_depth
            
            current_max_depth = max(current_max_depth, current_depth)
            
            if np.array_equal(Current_State, self.goal):
                self.create_path(Current_State_as_tuple)
                self.depth = current_max_depth
                self.Cost = len(self.Path) - 1
                return True
            
            if max_depth is not None and current_depth >= max_depth:
                continue
            
            neighbors = self.get_neighbors(Current_State)
            
            for neighbor in neighbors:
                new_tuple = self.matrix_to_tuple(neighbor)
                self.expanded_nodes.add(new_tuple)
                Ids_deque.append((neighbor, current_depth + 1, Current_State_as_tuple))
        
        self.depth = current_max_depth
        return False

    def create_path(self, goal_tuple):
        """Reconstruct the path from start to goal"""
        path = []
        current = goal_tuple
        start_tuple = self.matrix_to_tuple(self.matrix)
        
        while current != start_tuple:
            path.append(current)
            current = self.parent[current]
        
        path.append(start_tuple)
        path.reverse()
        
        self.Path = [np.array(state) for state in path]
        
        for i in range(len(self.Path) - 1):
            move = self.get_move_direction(self.Path[i], self.Path[i+1])
            self.moves.append(move)
        
        self.Cost = len(self.Path) - 1
        self.depth = len(self.Path) - 1

