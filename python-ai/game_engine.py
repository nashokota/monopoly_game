"""
Game Engine for AI Monopoly
Handles the main game loop and coordinates between agents
"""

import random
from typing import Tuple, List, Optional, Dict
from game_state import GameState, GAMBLE_EFFECTS


class GameEngine:
    """
    Main game engine that runs the Monopoly game between two AI agents.
    """
    
    def __init__(self, agent1, agent2, starting_cash: int = 1500, max_turns: int = 200):
        self.agents = [agent1, agent2]
        self.starting_cash = starting_cash
        self.max_turns = max_turns
        self.game_history: List[Dict] = []
        
    def reset_game(self) -> GameState:
        """Reset and start a new game"""
        state = GameState(
            cash=[self.starting_cash, self.starting_cash],
            max_turns=self.max_turns
        )
        self.game_history = []
        return state
    
    def play_turn(self, state: GameState) -> Tuple[GameState, Dict]:
        """
        Play a single turn of the game.
        Returns updated state and turn information.
        """
        current_player = state.current_player
        agent = self.agents[current_player]
        
        turn_info = {
            "turn": state.turn_count,
            "player": current_player,
            "agent": agent.get_name(),
            "initial_cash": state.cash.copy(),
            "initial_position": state.positions[current_player],
            "action": None,
            "dice_roll": 0,
            "new_position": 0,
            "landed_on": None,
            "fare_paid": 0,
            "property_bought": None,
            "gamble_effect": None,
        }
        
        # Roll dice
        dice_roll = state.roll_dice()
        state.last_dice_roll = dice_roll
        turn_info["dice_roll"] = dice_roll
        
        # Move player
        new_pos = state.move_player(current_player, dice_roll)
        turn_info["new_position"] = new_pos
        
        # Handle landing
        if state.is_gamble_tile(new_pos):
            # Gambling tile
            effect = random.choice(GAMBLE_EFFECTS)
            result = state.apply_gamble_effect(current_player, effect)
            state.last_gamble_effect = f"{effect.name}: {result}"
            turn_info["landed_on"] = "Gamble Tile"
            turn_info["gamble_effect"] = effect.name
        else:
            prop = state.get_property_at(new_pos)
            if prop:
                turn_info["landed_on"] = prop.name
                
                if prop.owner is not None and prop.owner != current_player:
                    # Pay fare
                    fare = state.pay_fare(current_player, prop)
                    turn_info["fare_paid"] = fare
                elif prop.owner is None:
                    # Can buy - ask agent for decision
                    action = agent.choose_action(state)
                    turn_info["action"] = action
                    state.last_action = action
                    
                    if action == "BUY":
                        state.buy_property(current_player, prop)
                        turn_info["property_bought"] = prop.name
                else:
                    turn_info["landed_on"] = f"{prop.name} (owned)"
        
        turn_info["final_cash"] = state.cash.copy()
        
        # Switch player
        state.current_player = 1 - current_player
        state.turn_count += 1
        
        # Check game over
        state.check_game_over()
        
        self.game_history.append(turn_info)
        
        return state, turn_info
    
    def play_game(self, verbose: bool = False) -> Tuple[GameState, int, List[Dict]]:
        """
        Play a complete game and return final state, winner, and history.
        """
        state = self.reset_game()
        
        while not state.game_over:
            state, turn_info = self.play_turn(state)
            
            if verbose:
                self._print_turn(turn_info)
        
        if verbose:
            self._print_game_result(state)
        
        return state, state.winner, self.game_history
    
    def _print_turn(self, info: Dict) -> None:
        """Print turn information"""
        print(f"\n--- Turn {info['turn']} ---")
        print(f"Player {info['player']} ({info['agent']})")
        print(f"Rolled: {info['dice_roll']}, Moved to position {info['new_position']}")
        print(f"Landed on: {info['landed_on']}")
        
        if info['action']:
            print(f"Action: {info['action']}")
        if info['property_bought']:
            print(f"Bought: {info['property_bought']}")
        if info['fare_paid']:
            print(f"Paid fare: ${info['fare_paid']}")
        if info['gamble_effect']:
            print(f"Gamble effect: {info['gamble_effect']}")
        
        print(f"Cash: P0=${info['final_cash'][0]}, P1=${info['final_cash'][1]}")
    
    def _print_game_result(self, state: GameState) -> None:
        """Print final game result"""
        print("\n" + "="*50)
        print("GAME OVER!")
        print("="*50)
        print(f"Winner: Player {state.winner} ({self.agents[state.winner].get_name()})")
        print(f"Total turns: {state.turn_count}")
        print(f"\nFinal Stats:")
        for i in range(2):
            print(f"\nPlayer {i} ({self.agents[i].get_name()}):")
            print(f"  Cash: ${state.cash[i]}")
            print(f"  Properties: {len(state.get_player_properties(i))}")
            print(f"  Property Value: ${state.get_player_property_value(i)}")
            print(f"  Total Wealth: ${state.cash[i] + state.get_player_property_value(i)}")


def run_tournament(agent1, agent2, num_games: int = 100, verbose: bool = False) -> Dict:
    """
    Run a tournament of multiple games between two agents.
    Returns statistics about the tournament.
    """
    results = {
        "agent1": agent1.get_name(),
        "agent2": agent2.get_name(),
        "total_games": num_games,
        "wins": [0, 0],
        "total_cash": [0, 0],
        "total_properties": [0, 0],
        "games": []
    }
    
    for game_num in range(num_games):
        # Alternate who goes first
        if game_num % 2 == 0:
            engine = GameEngine(agent1, agent2)
            player_map = [0, 1]
        else:
            engine = GameEngine(agent2, agent1)
            player_map = [1, 0]
        
        final_state, winner, history = engine.play_game(verbose=verbose)
        
        # Map winner back to original agent indices
        actual_winner = player_map[winner] if winner is not None else None
        
        if actual_winner is not None:
            results["wins"][actual_winner] += 1
        
        results["total_cash"][0] += final_state.cash[player_map.index(0)]
        results["total_cash"][1] += final_state.cash[player_map.index(1)]
        
        results["games"].append({
            "game_number": game_num + 1,
            "winner": actual_winner,
            "turns": final_state.turn_count
        })
        
        if (game_num + 1) % 10 == 0:
            print(f"Completed {game_num + 1}/{num_games} games...")
    
    results["win_rate"] = [w / num_games for w in results["wins"]]
    results["avg_cash"] = [t / num_games for t in results["total_cash"]]
    
    return results


def print_tournament_results(results: Dict) -> None:
    """Print tournament results"""
    print("\n" + "="*60)
    print("TOURNAMENT RESULTS")
    print("="*60)
    print(f"Games played: {results['total_games']}")
    print(f"\nAgent 0 ({results['agent1']}):")
    print(f"  Wins: {results['wins'][0]} ({results['win_rate'][0]*100:.1f}%)")
    print(f"  Average Cash: ${results['avg_cash'][0]:.0f}")
    print(f"\nAgent 1 ({results['agent2']}):")
    print(f"  Wins: {results['wins'][1]} ({results['win_rate'][1]*100:.1f}%)")
    print(f"  Average Cash: ${results['avg_cash'][1]:.0f}")
