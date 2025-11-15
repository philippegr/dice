"""
Visualization module for game solutions.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from solve_game import Solution


def plot_solution(
    board: list[list[int]],
    pieces_defs: list[list[list[int]]],
    solution: Solution
) -> plt.Figure:
    """
    Plot a single game solution visualization.
    
    Args:
        board: The game board as a 2D list (with -1 for empty cells)
        pieces_defs: Canonical piece definitions (list of 2D arrays)
        solution: Solution object containing piece and dice placements
        
    Returns:
        matplotlib.figure.Figure: The figure containing the visualization
    """
    n_rows = len(board)
    n_cols = len(board[0])
    
    # Create figure and axis - larger size for better rendering in Streamlit
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # Define colors for pieces
    piece_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive']
    
    # Create a grid to track which cells are covered by pieces
    piece_coverage = np.zeros((n_rows, n_cols), dtype=int)
    piece_colors_grid = np.zeros((n_rows, n_cols), dtype=int)
    
    # Mark cells covered by pieces
    for piece in solution.pieces:
        piece_idx = piece.piece_idx
        start_row, start_col = piece.position
        # Reconstruct piece shape from canonical definition and rotation
        piece_shape = np.rot90(np.array(pieces_defs[piece_idx]), piece.rotation)
        
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
                is_dice_cell = any(d.position == (i, j) for d in solution.dice)
                
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
                # Not bold when text is under a piece, bold otherwise
                if piece_coverage[i, j] == 1:
                    font_weight = 'normal'
                else:
                    font_weight = 'bold'
                ax.text(j + 0.5, n_rows - 1 - i + 0.5, str(cell_value),
                       ha='center', va='center', fontsize=12, 
                       fontweight=font_weight, color='black')
    
    # Set up the plot
    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.set_aspect('equal')
    ax.set_xticks(range(n_cols + 1))
    ax.set_yticks(range(n_rows + 1))
    ax.grid(True, alpha=0.3)
    ax.set_title('Game Solution\nRed=Vertical Bar, Blue=M, Green=W, Orange=Tetris, Purple=Weird, Brown=Not-L, Pink=Fat, Gray=Long, Olive=L\nLight Blue=Dice Cells', 
                 fontsize=10, pad=20)
    
    # Add legend for pieces
    legend_elements = []
    piece_names = ['Vertical Bar', 'M', 'W', 'Tetris', 'Weird', 'Not-L', 'Fat', 'Long', 'L']
    used_piece_indices = sorted(set(p.piece_idx for p in solution.pieces))
    for i, piece_idx in enumerate(used_piece_indices):
        if piece_idx < len(piece_names):
            legend_elements.append(
                patches.Patch(color=piece_colors[piece_idx % len(piece_colors)], 
                            label=f'Piece {piece_idx}: {piece_names[piece_idx]}')
            )
    
    legend_elements.append(patches.Patch(color='lightblue', label='Dice Cells'))
    legend_elements.append(patches.Patch(color='white', label='Empty Cells'))
    
    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    
    return fig

