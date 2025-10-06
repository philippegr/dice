# Barkmere Dice Game

A puzzle game solver that finds solutions for placing 9 different puzzle pieces on a 7x7 board to cover 6 dice positions.

## Features

- 🎲 Interactive web interface with Streamlit
- 🧩 Solves the puzzle using constraint programming (OR-Tools)
- 🎨 Visual solution display with colored pieces
- ⚡ Fast solving with symmetry breaking constraints

## Installation

```bash
# Install dependencies
uv sync

# Or manually install
uv add ortools matplotlib streamlit
```

## Usage

### Web Interface (Recommended)

```bash
# Run the Streamlit web app
uv run streamlit run app.py
```

Then open your browser to `http://localhost:8501`

### Command Line

```bash
# Run the standalone solver
uv run python solve_game.py

# Run the full enumeration solver
uv run python main.py
```

## How to Play

1. Open the web interface
2. Select 6 dice values (1-6) using the dropdowns
3. Click "Find Solution"
4. View the colored solution showing:
   - 🔵 Light blue: Dice positions
   - 🌈 Different colors: Different puzzle pieces
   - ⚪ White: Empty cells

## Game Rules

- Place all 9 puzzle pieces on the 7x7 board
- Each piece can be rotated (0°, 90°, 180°, 270°)
- Pieces 0 (Vertical Bar) and 3 (Tetris) have 2-fold symmetry
- Cover exactly 6 dice positions (one per dice value)
- All other cells must be covered by pieces

## Files

- `app.py` - Streamlit web interface
- `solve_game.py` - Standalone solver function
- `main.py` - Full enumeration solver with all solutions
- `pyproject.toml` - Project dependencies

## Example Solutions

The game has multiple solutions depending on dice values:
- `(6,6,6,6,6,6)` - 9 different solutions
- `(1,2,3,4,5,6)` - 1 solution
- Other combinations vary in solution count
