#!/usr/bin/env python3
"""
Script to enumerate all possible dice combinations and validate if a solution exists for each.
"""

import itertools
from solve_game import solve_game


def generate_all_dice_combinations():
    """
    Generate all unique dice combinations (multisets).
    
    Since solve_game sorts the dice values, we only need to consider multisets
    (combinations with repetition) rather than all ordered tuples.
    """
    return list(itertools.combinations_with_replacement(range(1, 7), 6))


def main():
    """Main function to enumerate and validate all dice combinations."""
    print("Enumerating all dice combinations...")
    all_combinations = generate_all_dice_combinations()
    total_combinations = len(all_combinations)
    
    unsolvable_combinations = []
    
    print(f"Testing {total_combinations} combinations...")
    for idx, combo in enumerate(all_combinations, 1):
        if idx % 50 == 0 or idx == total_combinations:
            print(f"Progress: {idx}/{total_combinations}")
        
        result = solve_game(combo, all_solutions=False)
        if len(result.solutions) == 0:
            unsolvable_combinations.append(combo)
    
    print()
    print(f"Number of unsolvable combinations: {len(unsolvable_combinations)}")
    print()
    
    if unsolvable_combinations:
        print("Unsolvable combinations:")
        for combo in unsolvable_combinations:
            print(f"  {combo}")
    else:
        print("All combinations are solvable!")


if __name__ == "__main__":
    main()

