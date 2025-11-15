from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

from ortools.sat.python import cp_model
from pandas.core.groupby import GroupBy

BOARD = [
    [1, 5, 4, 3, 2, 6, -1],
    [-1, 3, 1, 2, 6, 4, 5],
    [6, 2, 3, 5, 1, -1, 4],
    [3,4,2,6, -1, 5, 1], 
    [4, 6, -1, 1, 5, 2, 3],
    [5, -1, 6, 4, 3, 1, 2],
    [2, 1, 5, -1, 4, 3, 6]
]

pieces = [
    # Vertical bar
    [[1], 
    [1],
    [1]],
    # M
    [[1,1,1], 
    [1, 0, 1]], 
    # W 
    [[1,0,0],
    [1, 1, 0],
    [0,1,1]],
    # Tetris ? 
    [[1, 0, 0],
    [1,1,1],
    [0,0,1]],
    # Just weird 
    [[0,0,1],
    [1,1,1],
    [0,1,0]],
    # Not quite L
    [[0,1],
    [0,1],
    [0,1],
    [1,1]],
    # Fat
    [[1,1],
    [1,1],
    [1,0]], 
    # Long
    [[1,0],
    [1,1],
    [0,1],
    [0,1]],
    # L
    [[1,0],
    [1,0],
    [1,1]]
]

class MySolutionCallback(cp_model.CpSolverSolutionCallback):
    def __init__(self, piece_placements, dice_placements, board, original_pieces):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.piece_placements = piece_placements
        self.dice_placements = dice_placements
        self.board = board
        self.original_pieces = original_pieces
        self.solution_count = 0
        self.all_solutions = []

    def on_solution_callback(self):
        self.solution_count += 1
        
        # Extract the current solution
        solution_pieces = []
        solution_dice = []
        
        for placement in self.piece_placements:
            if self.Value(placement['variable']) == 1:
                solution_pieces.append(placement)
                
        for placement in self.dice_placements:
            if self.Value(placement['variable']) == 1:
                solution_dice.append(placement)
        
        solution_data = {
            'solution_number': self.solution_count,
            'pieces': solution_pieces,
            'dice': solution_dice,
            'board': self.board,
            'original_pieces': self.original_pieces
        }
        
        self.all_solutions.append(solution_data)
        print(f"Found solution {self.solution_count}")
        
        # Stop after finding a reasonable number of solutions to avoid infinite enumeration
        if self.solution_count >= 3000:
            self.StopSearch()


def solve(
    pieces: list[list[list[int]]], 
    board: list[list[int]], 
    dices = tuple[int, int, int, int, int, int]
):
    n_rows = len(board)
    n_cols = len(board[0])

    model = cp_model.CpModel()

    covered_positions = defaultdict(list)
    piece_placements = []  # Store piece placement info for visualization
    dice_placements = []   # Store dice placement info for visualization

    # Generate all possible positions for each piece
    for piece_idx, piece in enumerate(pieces):
        piece_rotations = []
        for r in range(4 if piece_idx not in {0,3} else 2): # bar and Tetris are symetrical 2 is enough
            rot = np.rot90(piece, r)

            width_rot = rot.shape[1]
            height_rot = rot.shape[0]
            for i in range(n_rows - height_rot + 1):
                for j in range(n_cols - width_rot + 1):
                    presence = model.NewBoolVar(f"piece_{piece_idx}_rot_{r}_at_{i}_{j}")
                    piece_rotations.append(presence)
                    
                    # Store placement info for visualization
                    piece_placement = {
                        'piece_idx': piece_idx,
                        'rotation': r,
                        'position': (i, j),
                        'piece_shape': rot,
                        'variable': presence
                    }
                    piece_placements.append(piece_placement)
                    
                    for ii in range(height_rot):
                        for jj in range(width_rot):
                            if rot[ii][jj] == 1:
                                covered_positions[(i + ii, j + jj)].append(presence)

        # Each piece must be placed
        model.add_exactly_one(piece_rotations)

    # Dice constraints
    dices = sorted(dices)
    for idx, dice_value in enumerate(dices):
        dice_positions = []
        for i in range(n_rows):
            for j in range(n_cols):
                if board[i][j] == dice_value:
                    is_dice_here = model.NewBoolVar(f"is_dice_here_{i}_{j}_{idx}")
                    dice_positions.append(is_dice_here)
                    covered_positions[(i, j)].append(is_dice_here)
                    
                    # Store dice placement info for visualization
                    dice_placement = {
                        'dice_value': dice_value,
                        'position': (i, j),
                        'variable': is_dice_here,
                        'idx': idx
                    }
                    dice_placements.append(dice_placement)
        # dice must be placed
        model.add_exactly_one(dice_positions)

        # symmetry breaking
        if idx > 0 and dices[idx-1]==dice_value:
            for i_end in range(0, n_rows):
                prevs = [p['variable'] for p in dice_placements if p['idx']==idx-1 and p['position'][0]<=i_end]
                current = [p['variable'] for p in dice_placements if p['idx']==idx and p['position'][0]<=i_end]
                model.add(sum(prevs) >= sum(current))
                      
    # Cells should be covered if not empty, otherwise, they can be empty
    for i in range(n_rows):
        for j in range(n_cols):
            #To debug 
            # model.add_at_most_one(covered_positions[i,j])
            if board[i][j] == -1:
               model.add_at_most_one(covered_positions[i,j])
            else:
               model.add_exactly_one(covered_positions[i,j])
    
    solver = cp_model.CpSolver()
    solver.parameters.log_search_progress = False  # Disable for cleaner output when finding multiple solutions
    solver.parameters.enumerate_all_solutions = True  # Enable enumeration of all solutions

    print("Searching for all feasible solutions...")
    solution_printer = MySolutionCallback(piece_placements, dice_placements, board, pieces)
    status = solver.Solve(model, solution_printer)

    print(f"\nStatus = {solver.StatusName(status)}")
    print(f"Number of solutions found: {solution_printer.solution_count}")
    
    return {
        'status': status,
        'solutions': solution_printer.all_solutions,
        'solution_count': solution_printer.solution_count,
        'board': board,
        'original_pieces': pieces
    }


def visualize_solution(solution_data, solution_number=1):
    """
    Visualize a single game solution with colored board showing piece placements and dice positions.
    
    Args:
        solution_data: Dictionary containing solution information from solve() function
        solution_number: Which solution to visualize (1-indexed)
    """
    if solution_data['status'] not in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        print(f"No solution found. Status: {solution_data['status']}")
        return
    
    if 'solutions' in solution_data:
        # Multiple solutions case
        if solution_number > len(solution_data['solutions']):
            print(f"Solution {solution_number} not found. Total solutions: {len(solution_data['solutions'])}")
            return
        solution = solution_data['solutions'][solution_number - 1]
        board = solution['board']
        pieces = solution['pieces']
        dice = solution['dice']
        solution_num = solution['solution_number']
    else:
        # Single solution case (backward compatibility)
        board = solution_data['board']
        pieces = solution_data['pieces']
        dice = solution_data['dice']
        solution_num = 1
    
    n_rows = len(board)
    n_cols = len(board[0])
    
    # Create figure and axis
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Define colors for pieces (different colors for each piece type)
    piece_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive']
    
    # Create a grid to track which cells are covered by pieces
    piece_coverage = np.zeros((n_rows, n_cols), dtype=int)
    piece_colors_grid = np.zeros((n_rows, n_cols), dtype=int)
    
    # Mark cells covered by pieces
    for piece in pieces:
        piece_idx = piece['piece_idx']
        start_row, start_col = piece['position']
        piece_shape = piece['piece_shape']
        
        for i in range(piece_shape.shape[0]):
            for j in range(piece_shape.shape[1]):
                if piece_shape[i, j] == 1:
                    row = start_row + i
                    col = start_col + j
                    piece_coverage[row, col] = 1
                    piece_colors_grid[row, col] = piece_idx
    
    # Create the main board visualization
    for i in range(n_rows):
        for j in range(n_cols):
            cell_value = board[i][j]
            
            # Check if this cell is covered by a piece
            if piece_coverage[i, j] == 1:
                # Color based on piece type
                color = piece_colors[piece_colors_grid[i, j] % len(piece_colors)]
            else:
                # Check if this cell has a dice
                is_dice_cell = False
                for d in dice:
                    if d['position'] == (i, j):
                        is_dice_cell = True
                        break
                
                if is_dice_cell:
                    # Light blue for dice cells
                    color = 'lightblue'
                else:
                    # White for empty cells
                    color = 'white'
            
            # Create rectangle for the cell
            rect = patches.Rectangle((j, n_rows - 1 - i), 1, 1, 
                                   linewidth=1, edgecolor='black', 
                                   facecolor=color)
            ax.add_patch(rect)
            
            # Add text for the cell value (except -1 which represents empty)
            if cell_value != -1:
                ax.text(j + 0.5, n_rows - 1 - i + 0.5, str(cell_value),
                       ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Set up the plot
    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.set_aspect('equal')
    ax.set_xticks(range(n_cols + 1))
    ax.set_yticks(range(n_rows + 1))
    ax.grid(True, alpha=0.3)
    ax.set_title(f'Game Solution Visualization - Solution {solution_num}\nRed=Vertical Bar, Blue=M, Green=W, Orange=Tetris, Purple=Weird, Brown=Not-L, Pink=Fat, Gray=Long, Olive=L\nLight Blue=Dice Cells', 
                 fontsize=10, pad=20)
    
    # Add legend for pieces
    legend_elements = []
    for i, color in enumerate(piece_colors[:len(pieces)]):
        piece_names = ['Vertical Bar', 'M', 'W', 'Tetris', 'Weird', 'Not-L', 'Fat', 'Long', 'L']
        if i < len(piece_names):
            legend_elements.append(patches.Patch(color=color, label=f'Piece {i+1}: {piece_names[i]}'))
    
    legend_elements.append(patches.Patch(color='lightblue', label='Dice Cells'))
    legend_elements.append(patches.Patch(color='white', label='Empty Cells'))
    
    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
    
    plt.tight_layout()
    
    # Save to subfolder instead of showing
    os.makedirs('solution_images', exist_ok=True)
    filename = f'solution_images/solution_{solution_num}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved visualization to: {filename}")
    plt.close()  # Close the figure to free memory
    
    # Print solution summary
    print(f"\nSolution {solution_num} Summary:")
    print(f"Status: {solution_data['status']}")
    print(f"Number of pieces placed: {len(pieces)}")
    print(f"Number of dice placed: {len(dice)}")
    
    print(f"\nPiece placements:")
    for i, piece in enumerate(pieces):
        print(f"  Piece {piece['piece_idx']}: Position {piece['position']}, Rotation {piece['rotation']}")
    
    print(f"\nDice placements:")
    for d in dice:
        print(f"  Dice {d['dice_value']}: Position {d['position']}")


def display_all_solutions(solution_data):
    """
    Display all found solutions with their details.
    
    Args:
        solution_data: Dictionary containing all solutions from solve() function
    """
    if solution_data['status'] not in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        print(f"No solutions found. Status: {solution_data['status']}")
        return
    
    print(f"\n{'='*60}")
    print(f"ALL SOLUTIONS SUMMARY")
    print(f"{'='*60}")
    print(f"Total solutions found: {solution_data['solution_count']}")
    
    for i, solution in enumerate(solution_data['solutions'], 1):
        print(f"\n--- Solution {i} ---")
        print(f"Piece placements:")
        for piece in solution['pieces']:
            print(f"  Piece {piece['piece_idx']}: Position {piece['position']}, Rotation {piece['rotation']}")
        
        print(f"Dice placements:")
        for d in solution['dice']:
            print(f"  Dice {d['dice_value']}: Position {d['position']}")
        
        # Check if this solution is identical to the previous one in terms of board coverage
        if i > 1:
            prev_solution = solution_data['solutions'][i-2]
            
            # Create board coverage maps for both solutions
            curr_coverage = {}
            prev_coverage = {}
            
            # Map each cell to which piece covers it
            for piece in solution['pieces']:
                piece_idx = piece['piece_idx']
                start_row, start_col = piece['position']
                piece_shape = piece['piece_shape']
                
                for r in range(piece_shape.shape[0]):
                    for c in range(piece_shape.shape[1]):
                        if piece_shape[r, c] == 1:
                            cell_pos = (start_row + r, start_col + c)
                            curr_coverage[cell_pos] = piece_idx
            
            for piece in prev_solution['pieces']:
                piece_idx = piece['piece_idx']
                start_row, start_col = piece['position']
                piece_shape = piece['piece_shape']
                
                for r in range(piece_shape.shape[0]):
                    for c in range(piece_shape.shape[1]):
                        if piece_shape[r, c] == 1:
                            cell_pos = (start_row + r, start_col + c)
                            prev_coverage[cell_pos] = piece_idx
            
            # Compare board coverage
            if curr_coverage == prev_coverage:
                print(f"  ⚠️  WARNING: Solution {i} has IDENTICAL board coverage to Solution {i-1}!")
            else:
                print(f"  ✅ Solution {i} has different board coverage from Solution {i-1}")
    
    # Save all solutions to images
    if solution_data['solution_count'] > 0:
        print(f"\n{'='*60}")
        print(f"SAVING ALL SOLUTIONS")
        print(f"{'='*60}")
        print(f"Saving all {solution_data['solution_count']} solutions to images...")
        
        for i in range(1, solution_data['solution_count'] + 1):
            print(f"Saving solution {i}...")
            visualize_solution(solution_data, i)
        
        print(f"\nAll solutions saved to 'solution_images/' folder!")


def main():
    print("Hello from game!")
    # Try with all dice as 6, which should have 8 solutions
    solution = solve(pieces=pieces, board=BOARD, dices=(1,2,3,4,5,6))
    #display_all_solutions(solution)


if __name__ == "__main__":
    main()
