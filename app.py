import streamlit as st
import matplotlib.pyplot as plt
from solve_game import solve_game
from visualize import plot_solution
from game_defs import BOARD, PIECES

# Page configuration
st.set_page_config(
    page_title="Barkmere Dice Game",
    page_icon="🎲",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Simple background
# st.markdown("""
# <style>
#     .stApp {
#         background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#     }
# </style>
# """, unsafe_allow_html=True)

# Main app
def main():
    # Header
    st.markdown("# 🎲 Barkmere Dice Game")
    st.markdown("Enter your 6 dice values to find the solution")
    
    # White container
    with st.container():
        st.markdown("### Select Your Dice Values")
        
        # Create two columns for dice inputs
        col1, col2, col3 = st.columns(3)
        
        with col1:
            dice1 = st.selectbox("Dice 1", options=["-", 1, 2, 3, 4, 5, 6], index=0)
            dice4 = st.selectbox("Dice 4", options=["-", 1, 2, 3, 4, 5, 6], index=0)
        
        with col2:
            dice2 = st.selectbox("Dice 2", options=["-", 1, 2, 3, 4, 5, 6], index=0)
            dice5 = st.selectbox("Dice 5", options=["-", 1, 2, 3, 4, 5, 6], index=0)
        
        with col3:
            dice3 = st.selectbox("Dice 3", options=["-", 1, 2, 3, 4, 5, 6], index=0)
            dice6 = st.selectbox("Dice 6", options=["-", 1, 2, 3, 4, 5, 6], index=0)
        
        # Solve button
        if st.button("Find Solution", type="primary"):
            # Check if all dice are selected
            if "-" in [dice1, dice2, dice3, dice4, dice5, dice6]:
                st.error("⚠️ Please select all dice values")
            else:
                # Create dice tuple
                dices = (dice1, dice2, dice3, dice4, dice5, dice6)
                
                # Show loading spinner
                with st.spinner("Solving puzzle..."):
                    # Solve the game
                    result = solve_game(dices)
                
                if len(result.solutions) > 0:
                    # Display the solution directly
                    st.markdown("### Solution")
                    fig = plot_solution(BOARD, PIECES, result.solutions[0])
                    st.pyplot(fig, width='stretch')
                    plt.close(fig)
                else:
                    st.error("❌ No Solution Found - Try different values!")

if __name__ == "__main__":
    main()
