"""
Expectiminimax Agent for AI Monopoly
Uses Minimax with chance nodes to handle dice randomness and gambling effects
"""

import random
from typing import Tuple, Optional, List
from game_state import GameState, GambleEffect, GAMBLE_EFFECTS, GAMBLE_POSITIONS

class ExpectiminimaxAgent:
    """
    Expectiminimax Agent - extends Minimax with chance nodes for stochastic elements.
    
    In this game, chance nodes appear:
    1. When rolling dice (2-12, non-uniform distribution)
    2. When landing on gambling tiles (random effects)
    
    Uses alpha-beta pruning where applicable and depth-limited search.
    """
    
    def __init__(self, player_id: int, max_depth: int = 4, use_sampling: bool = True, samples: int = 5):
        self.player_id = player_id
        self.max_depth = max_depth
        self.use_sampling = use_sampling  # Sample dice outcomes instead of all
        self.samples = samples
        self.nodes_evaluated = 0
        
        # Dice probabilities for 2d6
        self.dice_probs = self._calculate_dice_probabilities()
    
    def _calculate_dice_probabilities(self) -> dict:
        """Calculate probability distribution for 2d6"""
        probs = {}
        total = 36
        for i in range(1, 7):
            for j in range(1, 7):
                s = i + j
                probs[s] = probs.get(s, 0) + 1/total
        return probs
    
    def get_name(self) -> str:
        return f"Expectiminimax (depth={self.max_depth})"
    
    def choose_action(self, state: GameState) -> str:
        """
        Choose the best action (BUY or SKIP) for the current state.
        This is called after the dice roll and movement.
        """
        actions = state.get_available_actions(self.player_id)
        
        if len(actions) == 1:
            return actions[0]
        
        # Get property at current position
        pos = state.positions[self.player_id]
        prop = state.get_property_at(pos)
        
        # Quick heuristic: if we can afford and have enough cash buffer, lean towards buying
        if prop and prop.owner is None:
            cash_after = state.cash[self.player_id] - prop.price
            
            # Always buy if it completes a monopoly
            my_props = state.get_player_properties(self.player_id)
            same_color = [p for p in my_props if p.color == prop.color]
            total_same_color = len([p for p in state.properties if p.color == prop.color])
            
            if len(same_color) == total_same_color - 1:
                return "BUY"  # Complete the monopoly!
            
            # Buy if we have a reasonable cash buffer
            if cash_after >= 200:
                # Use deeper search only when it's a close decision
                pass  # Continue to minimax evaluation
            elif cash_after >= 100 and prop.price <= 150:
                return "BUY"  # Cheap property, go for it
        
        self.nodes_evaluated = 0
        best_action = "SKIP"
        best_value = float('-inf')
        
        for action in actions:
            # Simulate taking this action
            new_state = self._apply_action(state.copy(), self.player_id, action)
            
            # Evaluate using expectiminimax
            value = self._expectiminimax(
                new_state, 
                self.max_depth - 1, 
                False,  # Next is opponent's turn (minimizing)
                float('-inf'), 
                float('inf')
            )
            
            # Add immediate benefit of buying to counteract short-term cash loss
            if action == "BUY" and prop:
                # Buying has long-term value that shallow search might miss
                value += prop.fare * 2  # Expected fare income bonus
            
            if value > best_value:
                best_value = value
                best_action = action
        
        return best_action
    
    def _apply_action(self, state: GameState, player: int, action: str) -> GameState:
        """Apply an action to the state"""
        pos = state.positions[player]
        prop = state.get_property_at(pos)
        
        if action == "BUY" and prop:
            state.buy_property(player, prop)
        
        return state
    
    def _expectiminimax(self, state: GameState, depth: int, is_max: bool, 
                        alpha: float, beta: float) -> float:
        """
        Expectiminimax algorithm with alpha-beta pruning.
        
        Three types of nodes:
        - MAX: Maximizing player chooses best action
        - MIN: Minimizing player chooses best action
        - CHANCE: Average over random outcomes (dice/gambling)
        """
        self.nodes_evaluated += 1
        
        # Terminal conditions
        if depth == 0 or state.game_over:
            return state.evaluate(self.player_id)
        
        current_player = self.player_id if is_max else (1 - self.player_id)
        
        # First handle the chance node (dice roll)
        return self._chance_node(state, depth, is_max, current_player, alpha, beta)
    
    def _chance_node(self, state: GameState, depth: int, is_max: bool, 
                     player: int, alpha: float, beta: float) -> float:
        """
        Handle chance node - average over dice outcomes.
        """
        expected_value = 0.0
        
        if self.use_sampling:
            # Sample a subset of dice outcomes
            dice_outcomes = random.choices(
                list(self.dice_probs.keys()),
                weights=list(self.dice_probs.values()),
                k=self.samples
            )
            weight = 1.0 / len(dice_outcomes)
            
            for dice_roll in dice_outcomes:
                new_state = state.copy()
                new_pos = new_state.move_player(player, dice_roll)
                
                # Handle landing effects
                value = self._handle_landing(new_state, player, new_pos, depth, is_max, alpha, beta)
                expected_value += weight * value
        else:
            # Full expectation over all dice outcomes
            for dice_roll, prob in self.dice_probs.items():
                new_state = state.copy()
                new_pos = new_state.move_player(player, dice_roll)
                
                # Handle landing effects
                value = self._handle_landing(new_state, player, new_pos, depth, is_max, alpha, beta)
                expected_value += prob * value
        
        return expected_value
    
    def _handle_landing(self, state: GameState, player: int, position: int,
                        depth: int, is_max: bool, alpha: float, beta: float) -> float:
        """
        Handle what happens when landing on a tile.
        Returns the value after handling the landing.
        """
        # Check for gambling tile
        if state.is_gamble_tile(position):
            return self._gamble_chance_node(state, player, depth, is_max, alpha, beta)
        
        # Check for property
        prop = state.get_property_at(position)
        if prop:
            if prop.owner is not None and prop.owner != player:
                # Pay fare
                state.pay_fare(player, prop)
                state.current_player = 1 - player
                state.turn_count += 1
                state.check_game_over()
                return self._expectiminimax(state, depth - 1, not is_max, alpha, beta)
            elif prop.owner is None:
                # Decision node - can buy or skip
                return self._decision_node(state, player, depth, is_max, alpha, beta)
        
        # No special action needed
        state.current_player = 1 - player
        state.turn_count += 1
        state.check_game_over()
        return self._expectiminimax(state, depth - 1, not is_max, alpha, beta)
    
    def _gamble_chance_node(self, state: GameState, player: int, depth: int,
                            is_max: bool, alpha: float, beta: float) -> float:
        """
        Handle gambling tile - chance node over gambling effects.
        """
        expected_value = 0.0
        weight = 1.0 / len(GAMBLE_EFFECTS)
        
        for effect in GAMBLE_EFFECTS:
            new_state = state.copy()
            new_state.apply_gamble_effect(player, effect)
            new_state.current_player = 1 - player
            new_state.turn_count += 1
            new_state.check_game_over()
            
            value = self._expectiminimax(new_state, depth - 1, not is_max, alpha, beta)
            expected_value += weight * value
        
        return expected_value
    
    def _decision_node(self, state: GameState, player: int, depth: int,
                       is_max: bool, alpha: float, beta: float) -> float:
        """
        Decision node - player chooses to BUY or SKIP.
        """
        pos = state.positions[player]
        prop = state.get_property_at(pos)
        
        if not prop or prop.owner is not None:
            state.current_player = 1 - player
            state.turn_count += 1
            return self._expectiminimax(state, depth - 1, not is_max, alpha, beta)
        
        best_value = float('-inf') if is_max else float('inf')
        
        for action in ["BUY", "SKIP"]:
            new_state = state.copy()
            new_prop = new_state.get_property_at(pos)
            
            if action == "BUY" and new_state.cash[player] >= new_prop.price:
                new_state.buy_property(player, new_prop)
            
            new_state.current_player = 1 - player
            new_state.turn_count += 1
            new_state.check_game_over()
            
            value = self._expectiminimax(new_state, depth - 1, not is_max, alpha, beta)
            
            if is_max:
                best_value = max(best_value, value)
                alpha = max(alpha, value)
            else:
                best_value = min(best_value, value)
                beta = min(beta, value)
            
            if beta <= alpha:
                break  # Alpha-beta pruning
        
        return best_value
    
    def get_stats(self) -> dict:
        """Return agent statistics"""
        return {
            "name": self.get_name(),
            "player_id": self.player_id,
            "max_depth": self.max_depth,
            "nodes_evaluated": self.nodes_evaluated,
            "algorithm": "Expectiminimax with Alpha-Beta Pruning"
        }


class SimplifiedMinimaxAgent:
    """
    Simplified Minimax Agent - treats dice as average outcome.
    Faster but less accurate than full Expectiminimax.
    """
    
    def __init__(self, player_id: int, max_depth: int = 6):
        self.player_id = player_id
        self.max_depth = max_depth
        self.nodes_evaluated = 0
        self.average_dice = 7  # Average of 2d6
    
    def get_name(self) -> str:
        return f"Simplified Minimax (depth={self.max_depth})"
    
    def choose_action(self, state: GameState) -> str:
        """Choose the best action using simplified minimax"""
        actions = state.get_available_actions(self.player_id)
        
        if len(actions) == 1:
            return actions[0]
        
        self.nodes_evaluated = 0
        best_action = "SKIP"
        best_value = float('-inf')
        
        for action in actions:
            new_state = self._apply_action(state.copy(), self.player_id, action)
            value = self._minimax(new_state, self.max_depth - 1, False, float('-inf'), float('inf'))
            
            if value > best_value:
                best_value = value
                best_action = action
        
        return best_action
    
    def _apply_action(self, state: GameState, player: int, action: str) -> GameState:
        """Apply an action to the state"""
        pos = state.positions[player]
        prop = state.get_property_at(pos)
        
        if action == "BUY" and prop:
            state.buy_property(player, prop)
        
        return state
    
    def _minimax(self, state: GameState, depth: int, is_max: bool,
                 alpha: float, beta: float) -> float:
        """Standard minimax with alpha-beta pruning"""
        self.nodes_evaluated += 1
        
        if depth == 0 or state.game_over:
            return state.evaluate(self.player_id)
        
        current_player = self.player_id if is_max else (1 - self.player_id)
        
        # Simulate movement with average dice
        new_state = state.copy()
        new_pos = new_state.move_player(current_player, self.average_dice)
        
        # Handle gambling as average effect (net zero)
        if new_state.is_gamble_tile(new_pos):
            new_state.current_player = 1 - current_player
            new_state.turn_count += 1
            return self._minimax(new_state, depth - 1, not is_max, alpha, beta)
        
        prop = new_state.get_property_at(new_pos)
        
        if prop:
            if prop.owner is not None and prop.owner != current_player:
                new_state.pay_fare(current_player, prop)
                new_state.current_player = 1 - current_player
                new_state.turn_count += 1
                return self._minimax(new_state, depth - 1, not is_max, alpha, beta)
            elif prop.owner is None:
                # Decision: BUY or SKIP
                best = float('-inf') if is_max else float('inf')
                
                for action in ["BUY", "SKIP"]:
                    test_state = new_state.copy()
                    test_prop = test_state.get_property_at(new_pos)
                    
                    if action == "BUY" and test_state.cash[current_player] >= test_prop.price:
                        test_state.buy_property(current_player, test_prop)
                    
                    test_state.current_player = 1 - current_player
                    test_state.turn_count += 1
                    
                    val = self._minimax(test_state, depth - 1, not is_max, alpha, beta)
                    
                    if is_max:
                        best = max(best, val)
                        alpha = max(alpha, val)
                    else:
                        best = min(best, val)
                        beta = min(beta, val)
                    
                    if beta <= alpha:
                        break
                
                return best
        
        new_state.current_player = 1 - current_player
        new_state.turn_count += 1
        return self._minimax(new_state, depth - 1, not is_max, alpha, beta)
    
    def get_stats(self) -> dict:
        return {
            "name": self.get_name(),
            "player_id": self.player_id,
            "max_depth": self.max_depth,
            "nodes_evaluated": self.nodes_evaluated,
            "algorithm": "Simplified Minimax (Average Dice)"
        }
