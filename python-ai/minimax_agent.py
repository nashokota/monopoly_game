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
        Choose the best action (BUY, SKIP, or BUILD_X) for the current state.
        This is called after the dice roll and movement.
        """
        actions = state.get_available_actions(self.player_id)
        
        if len(actions) == 1:
            return actions[0]
        
        # Separate build actions from buy/skip actions
        build_actions = [a for a in actions if a.startswith("BUILD_")]
        base_actions = [a for a in actions if not a.startswith("BUILD_")]
        
        # Handle BUILD decisions - prioritize building when we have monopolies
        if build_actions and state.cash[self.player_id] > 300:
            # Find the best property to build on
            best_build = self._choose_best_build(state, build_actions)
            if best_build:
                return best_build
        
        # Get property at current position
        pos = state.positions[self.player_id]
        prop = state.get_property_at(pos)
        opponent = 1 - self.player_id
        
        if prop and prop.owner is None:
            cash_after = state.cash[self.player_id] - prop.price
            my_props = state.get_player_properties(self.player_id)
            opp_props = state.get_player_properties(opponent)
            
            # Count properties of same color
            same_color_mine = [p for p in my_props if p.color == prop.color]
            same_color_opp = [p for p in opp_props if p.color == prop.color]
            total_same_color = len([p for p in state.properties if p.color == prop.color])
            
            # ALWAYS buy if it completes a monopoly
            if len(same_color_mine) == total_same_color - 1:
                return "BUY"
            
            # ALWAYS buy to block opponent from completing monopoly
            if len(same_color_opp) >= total_same_color - 2:
                if cash_after >= 50:  # Buy even with low cash to block
                    return "BUY"
            
            # Strategic buying based on cash position
            # Minimum cash reserve based on opponent's potential fares
            max_opp_fare = max([p.fare for p in opp_props], default=0) * 2  # Could be doubled
            safe_reserve = max(100, max_opp_fare)
            
            # Aggressive early game (few properties sold)
            unsold = len(state.get_unowned_properties())
            if unsold > 25:  # Early game - be aggressive
                if cash_after >= 50:
                    return "BUY"
            elif unsold > 15:  # Mid game
                if cash_after >= safe_reserve:
                    return "BUY"
                elif prop.price <= 100 and cash_after >= 50:  # Cheap properties
                    return "BUY"
            else:  # Late game - be more conservative
                if cash_after >= safe_reserve * 1.5:
                    return "BUY"
        
        self.nodes_evaluated = 0
        best_action = "SKIP"
        best_value = float('-inf')
    
    def _choose_best_build(self, state: GameState, build_actions: list) -> Optional[str]:
        """Choose the best property to build on based on strategic value"""
        if not build_actions:
            return None
        
        best_action = None
        best_value = 0
        
        for action in build_actions:
            try:
                prop_index = int(action.split("_")[1])
                prop = state.get_property_at(prop_index)
                if prop:
                    # Value based on: fare increase, property location, current buildings
                    build_cost = state.get_building_cost(prop)
                    fare_increase = prop.fare * 0.4  # 20% of doubled fare
                    if state.has_monopoly(self.player_id, prop.color):
                        fare_increase *= 2  # Monopoly doubles base
                    
                    # Prefer properties with fewer buildings (more room to grow)
                    build_priority = (4 - prop.buildings) * 10
                    
                    # Prefer higher-value properties
                    value = fare_increase + build_priority + (prop.fare / 10)
                    
                    # Only build if we have enough cash reserve
                    if state.cash[self.player_id] - build_cost > 150 and value > best_value:
                        best_value = value
                        best_action = action
            except (ValueError, IndexError):
                continue
        
        return best_action
    
    def choose_asset_to_sell(self, state: GameState, amount_needed: int) -> List[dict]:
        """
        Intelligently choose which assets to sell when facing bankruptcy.
        Prioritizes selling:
        1. Buildings on low-value properties first
        2. Properties that don't contribute to monopolies
        3. Lower-value properties before higher-value ones
        Returns list of sell actions to perform in order.
        """
        sell_actions = []
        current_cash = state.cash[self.player_id]
        temp_state = state.copy()
        
        while current_cash < amount_needed:
            # First, try to sell buildings (prefer selling from low-value properties)
            sellable_buildings = temp_state.get_sellable_buildings(self.player_id)
            if sellable_buildings:
                # Score each building for selling (lower score = better to sell)
                best_to_sell = None
                best_score = float('inf')
                
                for prop in sellable_buildings:
                    score = self._calculate_building_keep_value(temp_state, prop)
                    if score < best_score:
                        best_score = score
                        best_to_sell = prop
                
                if best_to_sell:
                    sell_value = temp_state.get_sell_building_value(best_to_sell)
                    current_cash += sell_value
                    sell_actions.append({
                        "type": "SELL_BUILDING",
                        "property_index": best_to_sell.index,
                        "property_name": best_to_sell.name,
                        "value": sell_value
                    })
                    # Simulate the sale in temp state
                    temp_state.sell_building(self.player_id, best_to_sell)
                    continue
            
            # If no buildings to sell, sell properties
            sellable_props = temp_state.get_sellable_properties(self.player_id)
            if sellable_props:
                # Score each property for selling (lower score = better to sell)
                best_to_sell = None
                best_score = float('inf')
                
                for prop in sellable_props:
                    score = self._calculate_property_keep_value(temp_state, prop)
                    if score < best_score:
                        best_score = score
                        best_to_sell = prop
                
                if best_to_sell:
                    sell_value = temp_state.get_sell_property_value(best_to_sell)
                    current_cash += sell_value
                    sell_actions.append({
                        "type": "SELL_PROPERTY",
                        "property_index": best_to_sell.index,
                        "property_name": best_to_sell.name,
                        "value": sell_value
                    })
                    # Simulate the sale in temp state
                    temp_state.sell_property(self.player_id, best_to_sell)
                    continue
            
            # No more assets to sell
            break
        
        return sell_actions
    
    def _calculate_building_keep_value(self, state: GameState, prop) -> float:
        """
        Calculate how valuable it is to KEEP a building (higher = more valuable to keep).
        Lower score means we should sell this building first.
        """
        value = 0
        
        # Higher fare = more valuable to keep
        value += prop.fare * 2
        
        # If we have monopoly, buildings are more valuable
        if state.has_monopoly(self.player_id, prop.color):
            value += 200
        
        # Properties on expensive colors are more valuable
        value += prop.price / 2
        
        # More buildings on same property = each one less critical
        # (first building is most valuable, 4th is less critical)
        value -= (prop.buildings - 1) * 20
        
        return value
    
    def _calculate_property_keep_value(self, state: GameState, prop) -> float:
        """
        Calculate how valuable it is to KEEP a property (higher = more valuable to keep).
        Lower score means we should sell this property first.
        """
        value = 0
        opponent = 1 - self.player_id
        
        # Base value from fare potential
        value += prop.fare * 3
        
        # Check monopoly status
        player_props = state.get_player_properties(self.player_id)
        same_color = [p for p in player_props if p.color == prop.color]
        total_in_color = len([p for p in state.properties if p.color == prop.color])
        
        # If selling breaks a monopoly, HUGE penalty (don't sell!)
        if len(same_color) == total_in_color:
            value += 1000
        
        # If we have 3 out of 4 (close to monopoly), don't sell
        if len(same_color) == total_in_color - 1:
            value += 500
        
        # If opponent has other properties of same color, blocking value
        opp_props = state.get_player_properties(opponent)
        opp_same_color = [p for p in opp_props if p.color == prop.color]
        if opp_same_color:
            # This property blocks opponent's monopoly!
            value += 300 * len(opp_same_color)
        
        # Higher value properties are better to keep
        value += prop.price
        
        # Higher fare properties are better to keep
        value += prop.fare * 5
        
        return value
        
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
