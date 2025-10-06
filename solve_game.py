#!/usr/bin/env python3
"""
Standalone game solver that returns solution existence and visualization.
Usage: solve_game(dices) -> (bool, matplotlib_figure_or_None)
"""

from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ortools.sat.python import cp_model


# Game constants
PIECES = [
    [[1], [1], [1]],  # Vertical bar
    [[1,1,1], [1, 0, 1]],  # M
    [[1,0,0], [1, 1, 0], [0,1,1]],  # W 
    [[1, 0, 0], [1,1,1], [0,0,1]],  # Tetris
    [[0,0,1], [1,1,1], [0,1,0]],  # Weird
    [[0,1], [0,1], [0,1], [1,1]],  # Not-L
    [[1,1], [1,1], [0,1]],  # Fat
    [[1,0], [1,1], [0,1], [0,1]],  # Long
    [[1,0], [1,0], [1,1]]  # L
]

BOARD = [
    [1, 5, 4, 3, 2, 6, -1],
    [-1, 3, 1, 2, 6, 4, 5],
    [6, 2, 3, 5, 1, -1, 4],
    [3,4,2,6, -1, 5, 1], 
    [4, 6, -1, 1, 5, 2, 3],
    [5, -1, 6, 4, 3, 1, 2],
    [2, 1, 5, -1, 4, 3, 6]
]


def solve_game(dices):
    """
    Solve the game puzzle and return solution existence + visualization.
    
    Args:
        dices: Tuple of 6 dice values to place
        
    Returns:
        tuple: (solution_exists: bool, figure: matplotlib.figure.Figure or None)
    """
    n_rows = len(BOARD)
    n_cols = len(BOARD[0])
    
    model = cp_model.CpModel()
    covered_positions = defaultdict(list)
    piece_placements = []
    dice_placements = []
    
    # Generate all possible positions for each piece
    for piece_idx, piece in enumerate(PIECES):
        piece_rotations = []
        for r in range(4 if piece_idx not in {0,3} else 2):  # bar and Tetris are symmetrical
            rot = np.rot90(piece, r)
            
            width_rot = rot.shape[1]
            height_rot = rot.shape[0]
            for i in range(n_rows - height_rot + 1):
                for j in range(n_cols - width_rot + 1):
                    presence = model.NewBoolVar(f"piece_{piece_idx}_rot_{r}_at_{i}_{j}")
                    piece_rotations.append(presence)
                    
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
        
        model.add_exactly_one(piece_rotations)
    
    # Dice constraints
    for idx, dice_value in enumerate(dices):
        dice_positions = []
        for i in range(n_rows):
            for j in range(n_cols):
                if BOARD[i][j] == dice_value:
                    is_dice_here = model.NewBoolVar(f"is_dice_here_{i}_{j}_{idx}")
                    dice_positions.append(is_dice_here)
                    covered_positions[(i, j)].append(is_dice_here)
                    
                    dice_placement = {
                        'dice_value': dice_value,
                        'position': (i, j),
                        'variable': is_dice_here,
                        'idx': idx
                    }
                    dice_placements.append(dice_placement)
        
        model.add_exactly_one(dice_positions)
        
        # Symmetry breaking for dice
        if idx > 0 and dices[idx-1] == dice_value:
            for i_end in range(0, n_rows):
                prevs = [p['variable'] for p in dice_placements if p['idx']==idx-1 and p['position'][0]<=i_end]
                current = [p['variable'] for p in dice_placements if p['idx']==idx and p['position'][0]<=i_end]
                model.add(sum(prevs) >= sum(current))
    
    # Cell coverage constraints
    for i in range(n_rows):
        for j in range(n_cols):
            if BOARD[i][j] == -1:
                model.add_at_most_one(covered_positions[i,j])
            else:
                model.add_exactly_one(covered_positions[i,j])
    
    # Solve
    solver = cp_model.CpSolver()
    status = solver.solve(model)
    
    if status not in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        return False, None
    
    # Extract solution
    solution_pieces = []
    solution_dice = []
    
    for placement in piece_placements:
        if solver.value(placement['variable']) == 1:
            solution_pieces.append(placement)
    
    for placement in dice_placements:
        if solver.value(placement['variable']) == 1:
            solution_dice.append(placement)
    
    # Create visualization
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    piece_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive']
    piece_coverage = np.zeros((n_rows, n_cols), dtype=int)
    piece_colors_grid = np.zeros((n_rows, n_cols), dtype=int)
    
    # Mark cells covered by pieces
    for piece in solution_pieces:
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
    
    # Create the visualization
    for i in range(n_rows):
        for j in range(n_cols):
            cell_value = BOARD[i][j]
            
            if piece_coverage[i, j] == 1:
                color = piece_colors[piece_colors_grid[i, j] % len(piece_colors)]
            else:
                is_dice_cell = any(d['position'] == (i, j) for d in solution_dice)
                if is_dice_cell:
                    color = 'lightblue'
                else:
                    color = 'white'
            
            rect = patches.Rectangle((j, n_rows - 1 - i), 1, 1, 
                                   linewidth=1, edgecolor='black', 
                                   facecolor=color)
            ax.add_patch(rect)
            
            if cell_value != -1:
                ax.text(j + 0.5, n_rows - 1 - i + 0.5, str(cell_value),
                       ha='center', va='center', fontsize=12, fontweight='bold')
    
    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.set_aspect('equal')
    ax.set_xticks(range(n_cols + 1))
    ax.set_yticks(range(n_rows + 1))
    ax.grid(True, alpha=0.3)
    ax.set_title('Game Solution\nRed=Vertical Bar, Blue=M, Green=W, Orange=Tetris, Purple=Weird, Brown=Not-L, Pink=Fat, Gray=Long, Olive=L\nLight Blue=Dice Cells', 
                 fontsize=10, pad=20)
    
    # Add legend
    legend_elements = []
    for i, color in enumerate(piece_colors[:len(solution_pieces)]):
        piece_names = ['Vertical Bar', 'M', 'W', 'Tetris', 'Weird', 'Not-L', 'Fat', 'Long', 'L']
        if i < len(piece_names):
            legend_elements.append(patches.Patch(color=color, label=f'Piece {i}: {piece_names[i]}'))
    
    legend_elements.append(patches.Patch(color='lightblue', label='Dice Cells'))
    legend_elements.append(patches.Patch(color='white', label='Empty Cells'))
    
    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    
    return True, fig


# Example usage
if __name__ == "__main__":
    # Test with all 6s
    solution_exists, fig = solve_game((6,6,6,6,6,6))
    
    if solution_exists:
        print("✅ Solution found!")
        fig.savefig('solution.png', dpi=300, bbox_inches='tight')
        print("📁 Solution saved to 'solution.png'")
    else:
        print("❌ No solution found!")
