#!/usr/bin/env python3
"""
Standalone game solver that returns structured solution data.
Usage: solve_game(dices, all_solutions=False) -> SolveResult
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence
import numpy as np
from ortools.sat.python import cp_model
from game_defs import BOARD, PIECES


@dataclass(frozen=True)
class PiecePlacement:
    piece_idx: int
    rotation: int
    position: tuple[int, int]


@dataclass(frozen=True)
class DicePlacement:
    dice_value: int
    position: tuple[int, int]


@dataclass(frozen=True)
class Solution:
    pieces: Sequence[PiecePlacement]
    dice: Sequence[DicePlacement]


@dataclass(frozen=True)
class SolveResult:
    solutions: Sequence[Solution]


def _extract_solution(solver_or_callback, piece_placements, dice_placements):
    """
    Extract a solution from the solver/callback state.
    
    Args:
        solver_or_callback: Either a CpSolver or CpSolverSolutionCallback instance
        piece_placements: List of piece placement dictionaries
        dice_placements: List of dice placement dictionaries
        
    Returns:
        Solution: The extracted solution
    """
    pieces = [
        PiecePlacement(
            piece_idx=p['piece_idx'],
            rotation=p['rotation'],
            position=p['position'],
        )
        for p in piece_placements
        if solver_or_callback.BooleanValue(p['variable'])
    ]
    dice = [
        DicePlacement(
            dice_value=d['dice_value'],
            position=d['position'],
        )
        for d in dice_placements
        if solver_or_callback.BooleanValue(d['variable'])
    ]
    return Solution(pieces=pieces, dice=dice)


class _AllSolutionsCallback(cp_model.CpSolverSolutionCallback):
    def __init__(self, piece_placements, dice_placements):
        super().__init__()
        self._piece_placements = piece_placements
        self._dice_placements = dice_placements
        self.solution_count = 0
        self.solutions: list[Solution] = []

    def on_solution_callback(self):
        self.solution_count += 1
        solution = _extract_solution(self, self._piece_placements, self._dice_placements)
        self.solutions.append(solution)


# Game constants moved to game_defs.py


def solve_game(dices, all_solutions: bool = False):
    """
    Solve the game puzzle and return structured solution data.
    
    Args:
        dices: Tuple of 6 dice values to place
        all_solutions: If True, enumerate all feasible solutions. If False, return at most one solution.
        
    Returns:
        SolveResult: Contains a sequence of Solution objects
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
    dices = sorted(dices)
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
    if all_solutions:
        solver = cp_model.CpSolver()
        solver.parameters.enumerate_all_solutions = True
        solver.parameters.log_search_progress = False
        cb = _AllSolutionsCallback(piece_placements, dice_placements)
        solver.Solve(model, cb)
        return SolveResult(solutions=cb.solutions)
    
    # Single solution mode
    solver = cp_model.CpSolver()
    status = solver.Solve(model)
    
    if status not in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        return SolveResult(solutions=[])
    
    # Extract single solution
    single_solution = _extract_solution(solver, piece_placements, dice_placements)
    return SolveResult(solutions=[single_solution])


# Example usage
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from visualize import plot_solution
    from game_defs import BOARD, PIECES
    
    # Test with all 6s
    result = solve_game((6,6,6,6,6,6))
    
    if len(result.solutions) > 0:
        print(f"✅ Solution found! ({len(result.solutions)} solution(s))")
        # Visualize first solution
        fig = plot_solution(BOARD, PIECES, result.solutions[0])
        fig.savefig('solution.png', dpi=300, bbox_inches='tight')
        print("📁 Solution saved to 'solution.png'")
        plt.close(fig)
    else:
        print("❌ No solution found!")
