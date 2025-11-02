import sys
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QGridLayout, QPushButton, QLabel, 
                             QComboBox, QTextEdit, QGroupBox, QMessageBox,
                             QFrame, QSlider, QSpinBox)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QFont, QPalette, QColor, QLinearGradient, QPainter
import time
from BFS import BFS
from DFS import DFS
from IDS import IDS
from ASTAR import AStar , manhattan_distance , euclidean_distance
# Import your algorithms here
# from IterativeDFS import IterativeDFS
# from AStar import AStar, manhattan_distance, euclidean_distance


class GradientFrame(QFrame):
    """A frame with gradient background"""
    def __init__(self, color1="#1a1a2e", color2="#16213e", color3="#0f3460", parent=None):
        super().__init__(parent)
        self.color1 = color1
        self.color2 = color2
        self.color3 = color3
        
    def paintEvent(self, event):
        painter = QPainter(self)
        gradient = QLinearGradient(0, 0, 0, self.height())
        gradient.setColorAt(0.0, QColor(self.color1))
        gradient.setColorAt(0.5, QColor(self.color2))
        gradient.setColorAt(1.0, QColor(self.color3))
        painter.fillRect(self.rect(), gradient)


class ModernButton(QPushButton):
    """Modern styled button with hover effects"""
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setCursor(Qt.PointingHandCursor)
        self.setMinimumHeight(40)


class PuzzleBoard(QFrame):
    """Widget to display the 8-puzzle board"""
    def __init__(self):
        super().__init__()
        self.setFrameStyle(QFrame.Box | QFrame.Raised)
        self.setLineWidth(2)
        self.init_ui()
        
    def init_ui(self):
        self.layout = QGridLayout()
        self.layout.setSpacing(5)
        self.layout.setContentsMargins(10, 10, 10, 10)
        self.tiles = []
        
        # Create 3x3 grid of tiles
        for i in range(3):
            row = []
            for j in range(3):
                tile = QPushButton("")
                tile.setMinimumSize(80, 80)
                tile.setMaximumSize(80, 80)
                tile.setFont(QFont('Arial', 20, QFont.Bold))
                tile.setEnabled(False)
                self.layout.addWidget(tile, i, j)
                row.append(tile)
            self.tiles.append(row)
        
        self.setLayout(self.layout)
    
    def set_state(self, state):
        """Update the board with a new state"""
        for i in range(3):
            for j in range(3):
                value = int(state[i][j])
                if value == 0:
                    self.tiles[i][j].setText("")
                    self.tiles[i][j].setStyleSheet("""
                        QPushButton {
                            background-color: #1a1a2e;
                            border: 2px solid #0f3460;
                            border-radius: 10px;
                        }
                    """)
                else:
                    self.tiles[i][j].setText(str(value))
                    self.tiles[i][j].setStyleSheet("""
                        QPushButton {
                            background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                stop:0 #3498db, stop:1 #2980b9);
                            color: #ffffff;
                            border: 2px solid #1f618d;
                            border-radius: 10px;
                            font-weight: bold;
                        }
                        QPushButton:hover {
                            background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                stop:0 #3cb0fd, stop:1 #3498db);
                        }
                    """)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("8-Puzzle Solver - AI Assignment")
        self.setGeometry(100, 100, 1200, 700)
        
        # Solution data
        self.path = []
        self.path_to_goal = []
        self.current_step = 0
        self.timer = QTimer()
        self.timer.timeout.connect(self.show_next_step)
        
        self.init_ui()
        
    def init_ui(self):
        # Central widget with gradient background
        central_widget = GradientFrame("#1a1a2e", "#16213e", "#0f3460")
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(15, 15, 15, 15)
        
        # Left panel - Controls and Input
        left_panel = self.create_left_panel()
        main_layout.addWidget(left_panel, stretch=1)
        
        # Middle panel - Puzzle Display
        middle_panel = self.create_middle_panel()
        main_layout.addWidget(middle_panel, stretch=2)
        
        # Right panel - Results
        right_panel = self.create_right_panel()
        main_layout.addWidget(right_panel, stretch=1)
        
        self.apply_styles()
        
    def create_left_panel(self):
        """Create the left control panel"""
        panel = QGroupBox("Control Panel")
        layout = QVBoxLayout()
        layout.setSpacing(10)
        
        # Initial State Input
        init_group = QGroupBox("Initial State")
        init_layout = QVBoxLayout()
        init_layout.setSpacing(10)
        
        self.init_board = PuzzleBoard()
        init_layout.addWidget(self.init_board)
        
        # Manual input buttons
        input_layout = QGridLayout()
        input_layout.setSpacing(5)
        self.input_tiles = []
        for i in range(3):
            row = []
            for j in range(3):
                btn = ModernButton("0")
                btn.setMinimumSize(45, 45)
                btn.setFont(QFont('Arial', 12, QFont.Bold))
                btn.clicked.connect(lambda checked, x=i, y=j: self.cycle_tile(x, y))
                input_layout.addWidget(btn, i, j)
                row.append(btn)
            self.input_tiles.append(row)
        
        init_layout.addLayout(input_layout)
        
        # Button layout
        button_layout = QHBoxLayout()
        
        random_btn = ModernButton("🎲 Random")
        random_btn.clicked.connect(self.generate_random_state)
        button_layout.addWidget(random_btn)
        
        set_btn = ModernButton("✓ Set")
        set_btn.clicked.connect(self.set_initial_state)
        button_layout.addWidget(set_btn)
        
        init_layout.addLayout(button_layout)
        init_group.setLayout(init_layout)
        layout.addWidget(init_group)
        
        # Algorithm Selection
        algo_group = QGroupBox("Algorithm Selection")
        algo_layout = QVBoxLayout()
        
        self.algo_combo = QComboBox()
        self.algo_combo.addItems(["BFS", "DFS", "Iterative DFS", "A* (Manhattan)", "A* (Euclidean)"])
        algo_layout.addWidget(QLabel("Select Algorithm:"))
        algo_layout.addWidget(self.algo_combo)
        
        algo_group.setLayout(algo_layout)
        layout.addWidget(algo_group)
        
        # Solve Button
        self.solve_btn = ModernButton("🚀 SOLVE PUZZLE")
        self.solve_btn.setMinimumHeight(50)
        self.solve_btn.setFont(QFont('Arial', 14, QFont.Bold))
        self.solve_btn.clicked.connect(self.solve_puzzle)
        layout.addWidget(self.solve_btn)
        
        layout.addStretch()
        panel.setLayout(layout)
        return panel
    
    def create_middle_panel(self):
        """Create the middle panel with puzzle visualization"""
        panel = QGroupBox("Solution Visualization")
        layout = QVBoxLayout()
        layout.setSpacing(15)
        
        # Current state display
        self.current_board = PuzzleBoard()
        layout.addWidget(self.current_board, alignment=Qt.AlignCenter)
        
        # Step info
        self.step_label = QLabel("Step: 0 / 0")
        self.step_label.setFont(QFont('Arial', 16, QFont.Bold))
        self.step_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.step_label)
        
        # Move label
        self.move_label = QLabel("")
        self.move_label.setFont(QFont('Arial', 14))
        self.move_label.setAlignment(Qt.AlignCenter)
        self.move_label.setStyleSheet("color: #3498db; font-weight: bold;")
        layout.addWidget(self.move_label)
        
        # Control buttons
        control_layout = QHBoxLayout()
        control_layout.setSpacing(5)
        
        self.first_btn = ModernButton("⏮ First")
        self.first_btn.clicked.connect(self.show_first_step)
        self.first_btn.setEnabled(False)
        
        self.prev_btn = ModernButton("⏪ Previous")
        self.prev_btn.clicked.connect(self.show_previous_step)
        self.prev_btn.setEnabled(False)
        
        self.play_btn = ModernButton("▶ Play")
        self.play_btn.clicked.connect(self.toggle_play)
        self.play_btn.setEnabled(False)
        
        self.next_btn = ModernButton("Next ⏩")
        self.next_btn.clicked.connect(self.show_next_step)
        self.next_btn.setEnabled(False)
        
        self.last_btn = ModernButton("Last ⏭")
        self.last_btn.clicked.connect(self.show_last_step)
        self.last_btn.setEnabled(False)
        
        control_layout.addWidget(self.first_btn)
        control_layout.addWidget(self.prev_btn)
        control_layout.addWidget(self.play_btn)
        control_layout.addWidget(self.next_btn)
        control_layout.addWidget(self.last_btn)
        
        layout.addLayout(control_layout)
        
        # Speed control
        speed_layout = QHBoxLayout()
        speed_layout.addWidget(QLabel("Animation Speed:"))
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setMinimum(100)
        self.speed_slider.setMaximum(2000)
        self.speed_slider.setValue(500)
        self.speed_slider.setTickPosition(QSlider.TicksBelow)
        self.speed_slider.setTickInterval(300)
        speed_layout.addWidget(self.speed_slider)
        self.speed_label = QLabel("500 ms")
        self.speed_slider.valueChanged.connect(lambda v: self.speed_label.setText(f"{v} ms"))
        speed_layout.addWidget(self.speed_label)
        
        layout.addLayout(speed_layout)
        
        panel.setLayout(layout)
        return panel
    
    def create_right_panel(self):
        """Create the right panel with results"""
        panel = QGroupBox("Results & Statistics")
        layout = QVBoxLayout()
        layout.setSpacing(15)
        
        # Statistics
        stats_group = QGroupBox("Search Statistics")
        stats_layout = QVBoxLayout()
        stats_layout.setSpacing(8)
        
        self.cost_label = QLabel("Cost of Path: -")
        self.nodes_label = QLabel("Nodes Expanded: -")
        self.depth_label = QLabel("Search Depth: -")
        self.time_label = QLabel("Running Time: -")
        
        for label in [self.cost_label, self.nodes_label, self.depth_label, self.time_label]:
            label.setFont(QFont('Arial', 12))
            stats_layout.addWidget(label)
        
        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)
        
        # Path to goal
        path_group = QGroupBox("Solution Path")
        path_layout = QVBoxLayout()
        
        self.path_text = QTextEdit()
        self.path_text.setReadOnly(True)
        self.path_text.setMaximumHeight(120)
        path_layout.addWidget(self.path_text)
        
        path_group.setLayout(path_layout)
        layout.addWidget(path_group)
        
        # Goal State
        goal_group = QGroupBox("Goal State")
        goal_layout = QVBoxLayout()
        
        self.goal_board = PuzzleBoard()
        goal_state = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
        self.goal_board.set_state(goal_state)
        goal_layout.addWidget(self.goal_board)
        
        goal_group.setLayout(goal_layout)
        layout.addWidget(goal_group)
        
        layout.addStretch()
        panel.setLayout(layout)
        return panel
    
    def cycle_tile(self, i, j):
        """Cycle through numbers 0-8 when clicking input tile"""
        current = int(self.input_tiles[i][j].text())
        next_val = (current + 1) % 9
        self.input_tiles[i][j].setText(str(next_val))
    
    def generate_random_state(self):
        """Generate a solvable random initial state"""
        # Create a solved state and shuffle it
        state = list(range(9))
        
        # Ensure solvability by doing valid moves
        import random
        for _ in range(100):
            random.shuffle(state)
            if self.is_solvable(state):
                break
        
        # Update input tiles
        for i in range(3):
            for j in range(3):
                self.input_tiles[i][j].setText(str(state[i * 3 + j]))
    
    def is_solvable(self, puzzle):
        """Check if puzzle is solvable"""
        inversions = 0
        puzzle_list = [x for x in puzzle if x != 0]
        
        for i in range(len(puzzle_list)):
            for j in range(i + 1, len(puzzle_list)):
                if puzzle_list[i] > puzzle_list[j]:
                    inversions += 1
        
        return inversions % 2 == 0
    
    def set_initial_state(self):
        """Set the initial state from input tiles"""
        state = np.zeros((3, 3), dtype=int)
        values = []
        
        for i in range(3):
            for j in range(3):
                val = int(self.input_tiles[i][j].text())
                state[i][j] = val
                values.append(val)
        
        # Check if all numbers 0-8 are present
        if sorted(values) != list(range(9)):
            QMessageBox.warning(self, "Invalid State", "Please ensure all numbers 0-8 are used exactly once!")
            return
        
        # Check if solvable
        if not self.is_solvable(values):
            QMessageBox.warning(self, "Unsolvable State", "This configuration is not solvable!")
            return
        
        self.init_board.set_state(state)
        self.initial_state = state
        QMessageBox.information(self, "Success", "Initial state set successfully!")
    
    def solve_puzzle(self):
        """Solve the puzzle using selected algorithm"""
        if not hasattr(self, 'initial_state'):
            QMessageBox.warning(self, "No Initial State", "Please set an initial state first!")
            return
        
        algorithm = self.algo_combo.currentText()
        goal = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
        
        self.solve_btn.setEnabled(False)
        self.solve_btn.setText("Solving...")
        QApplication.processEvents()
        
        try:
            start_time = time.time()
            
            # Call the appropriate algorithm
            if algorithm == "BFS":
                solver = BFS(self.initial_state, goal)
                self.path, cost, nodes_expanded, depth = solver.BFS_Algorithm()
                self.path_to_goal = solver.path_to_goal
            
            elif algorithm == "DFS":
                    solver = DFS(self.initial_state, goal)
                    result = solver.DFS_Algorithm() 
                    
                    if result == 1:  # Success
                        self.path = solver.Path
                        cost = solver.Cost
                        nodes_expanded = len(solver.expanded_nodes)
                        depth = solver.depth
                        self.path_to_goal = solver.moves
                    else:
                        QMessageBox.information(self, "No Solution", "DFS could not find a solution!")
                        return  

            elif algorithm == "Iterative DFS":
                    solver = IDS(self.initial_state, goal)
                    result = solver.IDS_search()  
                    
                    if result == 1:  
                        self.path = solver.Path
                        cost = solver.Cost
                        nodes_expanded = len(solver.expanded_nodes)
                        depth = solver.depth
                        self.path_to_goal = solver.moves
                    else:
                        QMessageBox.information(self, "No Solution", "IDS could not find a solution!")
                        return
            
            elif "A*" in algorithm:
                if "Manhattan" in algorithm:
                    heuristic_func = manhattan_distance
                elif "Euclidean" in algorithm:
                    heuristic_func = euclidean_distance
                else:
                    heuristic_func = manhattan_distance  
                
                solver = AStar(self.initial_state, goal)
                self.path, cost, nodes_expanded, depth, self.path_to_goal = solver.AStar_Algorithm(heuristic_func)   
            end_time = time.time()
            
            self.display_results(cost, nodes_expanded, depth, end_time - start_time)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred: {str(e)}")
        
        finally:
            self.solve_btn.setEnabled(True)
            self.solve_btn.setText("🚀 SOLVE PUZZLE")
    
    def simulate_solution(self):
        """Simulate a solution for testing (remove when you integrate real algorithms)"""
        # Create a dummy solution path
        self.path = [
            self.initial_state,
            self.initial_state.copy(),
            np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
        ]
        self.path_to_goal = ["Up", "Left"]
        
        cost = len(self.path_to_goal)
        nodes_expanded = 50
        depth = cost
        running_time = 0.05
        
        self.display_results(cost, nodes_expanded, depth, running_time)
    
    def display_results(self, cost, nodes_expanded, depth, running_time):
        """Display the results of the search"""
        self.cost_label.setText(f"Cost of Path: {cost}")
        self.nodes_label.setText(f"Nodes Expanded: {nodes_expanded}")
        self.depth_label.setText(f"Search Depth: {depth}")
        self.time_label.setText(f"Running Time: {running_time:.4f} seconds")
        
        # Display path to goal
        if self.path_to_goal:
            path_str = " → ".join(self.path_to_goal)
            self.path_text.setText(path_str)
        else:
            self.path_text.setText("No solution found!")
        
        # Enable visualization controls
        if len(self.path) > 0:
            self.current_step = 0
            self.show_current_step()
            self.first_btn.setEnabled(True)
            self.prev_btn.setEnabled(True)
            self.play_btn.setEnabled(True)
            self.next_btn.setEnabled(True)
            self.last_btn.setEnabled(True)
    
    def show_current_step(self):
        """Display the current step"""
        if 0 <= self.current_step < len(self.path):
            self.current_board.set_state(self.path[self.current_step])
            self.step_label.setText(f"Step: {self.current_step} / {len(self.path) - 1}")
            
            # Show move
            if self.current_step > 0 and self.current_step - 1 < len(self.path_to_goal):
                self.move_label.setText(f"Move: {self.path_to_goal[self.current_step - 1]}")
            else:
                self.move_label.setText("Start State" if self.current_step == 0 else "Goal Reached!")
    
    def show_first_step(self):
        """Show the first step"""
        self.current_step = 0
        self.show_current_step()
    
    def show_previous_step(self):
        """Show the previous step"""
        if self.current_step > 0:
            self.current_step -= 1
            self.show_current_step()
    
    def show_next_step(self):
        """Show the next step"""
        if self.current_step < len(self.path) - 1:
            self.current_step += 1
            self.show_current_step()
        else:
            self.timer.stop()
            self.play_btn.setText("▶ Play")
    
    def show_last_step(self):
        """Show the last step"""
        self.current_step = len(self.path) - 1
        self.show_current_step()
    
    def toggle_play(self):
        """Toggle auto-play of solution"""
        if self.timer.isActive():
            self.timer.stop()
            self.play_btn.setText("▶ Play")
        else:
            if self.current_step >= len(self.path) - 1:
                self.current_step = 0
            self.timer.start(self.speed_slider.value())
            self.play_btn.setText("⏸ Pause")
    
    def apply_styles(self):
        """Apply global styles to the application - Modern dark theme with blue accents"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1a1a2e;
            }
            QWidget {
                background-color: #1a1a2e;
                color: #e6e6e6;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #3498db;
                border-radius: 10px;
                margin-top: 15px;
                padding-top: 15px;
                background-color: rgba(30, 30, 46, 180);
                color: #3498db;
                font-size: 14px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 10px 0 10px;
                color: #3498db;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton {
                background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #3498db, stop:1 #2980b9);
                color: #ffffff;
                border: none;
                padding: 10px 15px;
                border-radius: 8px;
                font-size: 13px;
                font-weight: bold;
                min-height: 30px;
            }
            QPushButton:hover {
                background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #3cb0fd, stop:1 #3498db);
            }
            QPushButton:pressed {
                background-color: #21618c;
            }
            QPushButton:disabled {
                background-color: #2c3e50;
                color: #7f8c8d;
            }
            QComboBox {
                padding: 8px 15px;
                border: 2px solid #3498db;
                border-radius: 8px;
                background-color: #1a1a2e;
                color: #e6e6e6;
                selection-background-color: #3498db;
                font-size: 13px;
                min-height: 30px;
            }
            QComboBox::drop-down {
                border: none;
                background-color: #3498db;
                width: 30px;
                border-radius: 5px;
            }
            QComboBox::down-arrow {
                image: none;
                border: 2px solid #ffffff;
                width: 8px;
                height: 8px;
                border-width: 0 3px 3px 0;
                transform: rotate(45deg);
            }
            QComboBox QAbstractItemView {
                background-color: #1a1a2e;
                color: #e6e6e6;
                selection-background-color: #3498db;
                selection-color: #ffffff;
                border: 2px solid #3498db;
                border-radius: 5px;
            }
            QTextEdit {
                border: 2px solid #3498db;
                border-radius: 8px;
                padding: 10px;
                background-color: #1a1a2e;
                color: #e6e6e6;
                font-size: 12px;
                selection-background-color: #3498db;
            }
            QLabel {
                color: #e6e6e6;
                font-size: 13px;
            }
            QSlider::groove:horizontal {
                border: 1px solid #3498db;
                height: 8px;
                background: #1a1a2e;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #3498db;
                border: 2px solid #3cb0fd;
                width: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }
            QSlider::handle:horizontal:hover {
                background: #3cb0fd;
            }
            QFrame {
                background-color: rgba(30, 30, 46, 180);
                border: 2px solid #3498db;
                border-radius: 10px;
            }
            QMessageBox {
                background-color: #1a1a2e;
                color: #e6e6e6;
            }
            QMessageBox QPushButton {
                min-width: 80px;
            }
        """)


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    # Set dark palette for message boxes
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(26, 26, 46))
    palette.setColor(QPalette.WindowText, QColor(230, 230, 230))
    palette.setColor(QPalette.Base, QColor(30, 30, 46))
    palette.setColor(QPalette.AlternateBase, QColor(26, 26, 46))
    palette.setColor(QPalette.Text, QColor(230, 230, 230))
    palette.setColor(QPalette.Button, QColor(52, 152, 219))
    palette.setColor(QPalette.ButtonText, QColor(255, 255, 255))
    palette.setColor(QPalette.Highlight, QColor(52, 152, 219))
    palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
    app.setPalette(palette)
    
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()