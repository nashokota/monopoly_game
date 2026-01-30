"""
Test script to verify AI agents work correctly
"""

from game_state import GameState
from minimax_agent import ExpectiminimaxAgent, SimplifiedMinimaxAgent
from mcts_agent import MCTSAgent, HybridMCTSAgent
from game_engine import GameEngine, run_tournament, print_tournament_results

def test_game_state():
    """Test basic game state functionality"""
    print("Testing GameState...")
    
    state = GameState()
    
    # Check board creation
    assert len(state.board) == 40, "Board should have 40 tiles"
    assert len(state.properties) == 36, "Should have 36 properties"
    assert len(state.gamble_tiles) == 4, "Should have 4 gamble tiles"
    
    # Check starting positions
    assert state.positions == [0, 0], "Players should start at position 0"
    assert state.cash == [1500, 1500], "Players should start with $1500"
    
    # Test dice roll
    for _ in range(100):
        roll = state.roll_dice()
        assert 2 <= roll <= 12, f"Dice roll {roll} out of range"
    
    # Test movement
    state.move_player(0, 5)
    assert state.positions[0] == 5, "Player should be at position 5"
    
    state.move_player(0, 38)  # Test wrapping
    assert state.positions[0] == 3, "Player should wrap around to position 3"
    
    # Test property buying
    prop = state.get_property_at(3)
    if prop:
        initial_cash = state.cash[0]
        state.buy_property(0, prop)
        assert prop.owner == 0, "Player 0 should own the property"
        assert state.cash[0] == initial_cash - prop.price, "Cash should decrease"
    
    print("✓ GameState tests passed!")

def test_minimax_agent():
    """Test Expectiminimax agent"""
    print("\nTesting Expectiminimax Agent...")
    
    agent = ExpectiminimaxAgent(player_id=0, max_depth=3)
    state = GameState()
    
    # Move to a property
    state.move_player(0, 2)
    
    # Get action
    action = agent.choose_action(state)
    assert action in ["BUY", "SKIP"], f"Invalid action: {action}"
    
    print(f"  Agent chose: {action}")
    print(f"  Nodes evaluated: {agent.nodes_evaluated}")
    print("✓ Expectiminimax tests passed!")

def test_mcts_agent():
    """Test MCTS agent"""
    print("\nTesting MCTS Agent...")
    
    agent = MCTSAgent(player_id=0, num_simulations=100)
    state = GameState()
    
    # Move to a property
    state.move_player(0, 2)
    
    # Get action
    action = agent.choose_action(state)
    assert action in ["BUY", "SKIP"], f"Invalid action: {action}"
    
    print(f"  Agent chose: {action}")
    print(f"  Total simulations: {agent.total_simulations}")
    print("✓ MCTS tests passed!")

def test_single_game():
    """Test playing a single game"""
    print("\nTesting single game...")
    
    agent1 = ExpectiminimaxAgent(player_id=0, max_depth=2)
    agent2 = MCTSAgent(player_id=1, num_simulations=50)
    
    engine = GameEngine(agent1, agent2, max_turns=50)
    final_state, winner, history = engine.play_game(verbose=False)
    
    print(f"  Game completed in {final_state.turn_count} turns")
    print(f"  Winner: Player {winner}")
    print(f"  Final cash: P0=${final_state.cash[0]}, P1=${final_state.cash[1]}")
    print("✓ Single game test passed!")

def test_tournament():
    """Test running a small tournament"""
    print("\nTesting tournament (5 games)...")
    
    agent1 = ExpectiminimaxAgent(player_id=0, max_depth=2)
    agent2 = MCTSAgent(player_id=1, num_simulations=50)
    
    results = run_tournament(agent1, agent2, num_games=5, verbose=False)
    
    print(f"  {results['agent1']}: {results['wins'][0]} wins")
    print(f"  {results['agent2']}: {results['wins'][1]} wins")
    print("✓ Tournament test passed!")

if __name__ == "__main__":
    print("="*50)
    print("AI Monopoly - Test Suite")
    print("="*50)
    
    test_game_state()
    test_minimax_agent()
    test_mcts_agent()
    test_single_game()
    test_tournament()
    
    print("\n" + "="*50)
    print("All tests passed! ✓")
    print("="*50)
