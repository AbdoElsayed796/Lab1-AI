from Node import Node
import numpy as np

class Tree:
    def __init__(self, root_value):
        self.root = Node(root_value)

    def add(self, parent_value, child_value):
        parent_node = self.find(self.root, parent_value)
        if parent_node:
            parent_node.add_child(Node(child_value))
        else:
            print(f"Parent not found!")

    def find(self, node, value):
        if np.array_equal(node.value, value):
            return node
        for child in node.children:
            found = self.find(child, value)
            if found:
                return found
        return None
        
    def print_tree(self, node=None, level=0):
        if node is None:
            node = self.root
        
        indent = "  " * level
        print(f"{indent}Level {level}:")
        print(f"{indent}{node.value}")
        
        for child in node.children:
            self.print_tree(child, level + 1)
    
    def count_tree_nodes(self, node=None):
        if node is None:
            node = self.root
        
        count = 1
        for child in node.children:
            count += self.count_tree_nodes(child)
        return count