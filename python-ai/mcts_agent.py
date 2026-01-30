"""
Enhanced Monte Carlo Tree Search (MCTS) Agent for AI Monopoly
Powerful simulation-based approach with strategic heuristics to compete with Expectiminimax
"""

import random
import math
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, field
from game_state import GameState, GAMBLE_EFFECTS

@dataclass
class MCTSNode:
    """Node in the MCTS tree with enhanced statistics"""
    state: GameState
    parent: Optional['MCTSNode'] = None
    action: Optional[str] = None
    children: List['MCTSNode'] = field(default_factory=list)
    visits: int = 0
    total_reward: float = 0.0
    squared_reward: float = 0.0  # For variance calculation
    untried_actions: List[str] = field(default_factory=list)
    player: int = 0
    prior_value: float = 0.5  # Prior evaluation for this action
    
    def ucb1_tuned(self, exploration_constant: float = 1.0) -> float:
        """
        UCB1-Tuned: Better exploration using variance estimates
        """
        if self.visits == 0:
            return float('inf')
        
        mean_reward = self.total_reward / self.visits
        
        # Variance estimate
        variance = (self.squared_reward / self.visits) - (mean_reward ** 2)
        variance = max(0, variance)  # Numerical stability
        
        # UCB1-Tuned exploration term
        log_parent = math.log(self.parent.visits)
        exploration_term = min(0.25, variance + math.sqrt(2 * log_parent / self.visits))
        
        ucb = mean_reward + exploration_constant * math.sqrt(log_parent / self.visits * exploration_term)
        
        # Add prior bonus for less-visited nodes
        prior_bonus = self.prior_value * math.sqrt(self.parent.visits) / (1 + self.visits)
        
        return ucb + 0.5 * prior_bonus
    
    def best_child(self, exploration_constant: float = 1.0) -> 'MCTSNode':
        """Select best child based on UCB1-Tuned"""
        return max(self.children, key=lambda c: c.ucb1_tuned(exploration_constant))
    
    def is_fully_expanded(self) -> bool:
        return len(self.untried_actions) == 0
    
    def is_terminal(self) -> bool:
        return self.state.game_over


class MCTSAgent:
    """
    Enhanced Monte Carlo Tree Search Agent
    
    Improvements over basic MCTS:
    1. UCB1-Tuned for better exploration
    2. Prior knowledge from heuristics
    3. Smart simulation policy with strategic buying
    4. RAVE (Rapid Action Value Estimation) inspired enhancements
    5. Monopoly-aware evaluation
    """
    
    def __init__(self, player_id: int, num_simulations: int = 2000, 
                 exploration_constant: float = 1.0, max_simulation_depth: int = 60):
        self.player_id = player_id
        self.num_simulations = num_simulations
        self.exploration_constant = exploration_constant
        self.max_simulation_depth = max_simulation_depth
        self.total_simulations = 0
        
        # Learning: track action values across games
        self.action_stats = {"BUY": {"wins": 0, "games": 0}, "SKIP": {"wins": 0, "games": 0}, "BUILD": {"wins": 0, "games": 0}}
    
    def get_name(self) -> str:
        return f"MCTS (sims={self.num_simulations})"
    
    def choose_action(self, state: GameState) -> str:
        """
        Choose the best action using enhanced MCTS with strategic overrides.
        """
        actions = state.get_available_actions(self.player_id)
        
        if len(actions) == 1:
            return actions[0]
        
        # Separate build actions from buy/skip actions
        build_actions = [a for a in actions if a.startswith("BUILD_")]
        base_actions = [a for a in actions if not a.startswith("BUILD_")]
        
        # Check for build opportunities first
        if build_actions and state.cash[self.player_id] > 300:
            best_build = self._choose_best_build(state, build_actions)
            if best_build:
                return best_build
        
        # Strategic overrides for critical decisions
        override = self._check_strategic_override(state)
        if override:
            return override
        
        # Create root node with prior values (only for base actions)
        root = MCTSNode(
            state=state.copy(),
            untried_actions=base_actions.copy(),
            player=self.player_id
        )
        
        # Set prior values for actions
        self._set_action_priors(root, state)
        
        # Run simulations
        for _ in range(self.num_simulations):
            self.total_simulations += 1
            
            # Selection & Expansion
            node = self._select(root)
            
            # Simulation with smart policy
            reward = self._simulate(node.state, node.player)
            
            # Backpropagation
            self._backpropagate(node, reward)
        
        # Choose best action (most visited with value consideration)
        if not root.children:
            return random.choice(base_actions) if base_actions else "SKIP"
        
        # Use a combination of visits and value for final selection
        best_child = max(root.children, key=lambda c: c.visits + c.total_reward)
        
        # Update learning stats
        self._update_action_stats(best_child.action, state)
        
        return best_child.action
    
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
    
    def _check_strategic_override(self, state: GameState) -> Optional[str]:
        """
        Critical strategic decisions that should override MCTS.
        These are high-confidence heuristics.
        """
        pos = state.positions[self.player_id]
        prop = state.get_property_at(pos)
        
        if not prop or prop.owner is not None:
            return None
        
        opponent = 1 - self.player_id
        cash = state.cash[self.player_id]
        cash_after = cash - prop.price
        
        my_props = state.get_player_properties(self.player_id)
        opp_props = state.get_player_properties(opponent)
        
        same_color_mine = [p for p in my_props if p.color == prop.color]
        same_color_opp = [p for p in opp_props if p.color == prop.color]
        total_same_color = len([p for p in state.properties if p.color == prop.color])
        
        # CRITICAL: Complete monopoly - always buy
        if len(same_color_mine) == total_same_color - 1 and cash >= prop.price:
            return "BUY"
        
        # CRITICAL: Block opponent monopoly - almost always buy
        if len(same_color_opp) == total_same_color - 1 and cash >= prop.price:
            if cash_after >= 30:  # Only need minimal reserve
                return "BUY"
        
        # Strong block: opponent has n-2, we have 0
        if len(same_color_opp) >= total_same_color - 2 and len(same_color_mine) == 0:
            if cash_after >= 50:
                return "BUY"
        
        # Early game aggressive buying
        unsold = len(state.get_unowned_properties())
        if unsold > 28 and cash >= prop.price:
            if cash_after >= 100:
                return "BUY"
        
        # Good value property (high fare relative to price)
        fare_ratio = prop.fare / prop.price
        if fare_ratio >= 0.45 and cash_after >= 150:
            return "BUY"
        
        return None
    
    def _set_action_priors(self, root: MCTSNode, state: GameState):
        """
        Set prior values for actions based on heuristics.
        This guides initial exploration.
        """
        pos = state.positions[self.player_id]
        prop = state.get_property_at(pos)
        
        if not prop or prop.owner is not None:
            return
        
        buy_prior = 0.5
        skip_prior = 0.5
        
        opponent = 1 - self.player_id
        cash = state.cash[self.player_id]
        cash_after = cash - prop.price
        
        my_props = state.get_player_properties(self.player_id)
        opp_props = state.get_player_properties(opponent)
        
        same_color_mine = [p for p in my_props if p.color == prop.color]
        same_color_opp = [p for p in opp_props if p.color == prop.color]
        total_same_color = len([p for p in state.properties if p.color == prop.color])
        
        # Adjust priors based on situation
        
        # Strong buy signals
        if len(same_color_mine) >= 1:
            buy_prior += 0.2  # Build on existing set
        
        if len(same_color_opp) >= 2:
            buy_prior += 0.3  # Block opponent
        
        # Cash considerations
        if cash_after < 50:
            skip_prior += 0.3  # Low cash - risky to buy
        elif cash_after > 300:
            buy_prior += 0.2  # Plenty of cash
        
        # Game phase
        unsold = len(state.get_unowned_properties())
        if unsold > 25:  # Early game
            buy_prior += 0.1
        elif unsold < 10:  # Late game
            skip_prior += 0.1
        
        # Property value
        fare_ratio = prop.fare / prop.price
        if fare_ratio >= 0.45:
            buy_prior += 0.15
        
        # Normalize
        total = buy_prior + skip_prior
        root.prior_value = buy_prior / total
    
    def _select(self, node: MCTSNode) -> MCTSNode:
        """
        Selection phase with UCB1-Tuned
        """
        current = node
        
        while not current.is_terminal():
            if not current.is_fully_expanded():
                return self._expand(current)
            else:
                if current.children:
                    current = current.best_child(self.exploration_constant)
                else:
                    break
        
        return current
    
    def _expand(self, node: MCTSNode) -> MCTSNode:
        """
        Expansion phase with prior value initialization
        """
        if not node.untried_actions:
            return node
        
        # Prefer trying "BUY" first if it's available and not yet tried
        action = node.untried_actions.pop()
        
        # Create new state
        new_state = node.state.copy()
        pos = new_state.positions[node.player]
        prop = new_state.get_property_at(pos)
        
        if action == "BUY" and prop and new_state.cash[node.player] >= prop.price:
            new_state.buy_property(node.player, prop)
        
        # Complete turn
        new_state = self._simulate_turn_completion(new_state, node.player)
        
        next_player = new_state.current_player
        next_actions = new_state.get_available_actions(next_player)
        
        # Calculate prior for this action
        prior = self._calculate_action_prior(node.state, node.player, action)
        
        child = MCTSNode(
            state=new_state,
            parent=node,
            action=action,
            untried_actions=next_actions,
            player=next_player,
            prior_value=prior
        )
        
        node.children.append(child)
        return child
    
    def _calculate_action_prior(self, state: GameState, player: int, action: str) -> float:
        """Calculate prior value for an action"""
        if action == "SKIP":
            return 0.4
        
        # BUY action
        pos = state.positions[player]
        prop = state.get_property_at(pos)
        
        if not prop:
            return 0.5
        
        opponent = 1 - player
        my_props = state.get_player_properties(player)
        opp_props = state.get_player_properties(opponent)
        
        same_color_mine = len([p for p in my_props if p.color == prop.color])
        same_color_opp = len([p for p in opp_props if p.color == prop.color])
        
        prior = 0.5
        if same_color_mine >= 1:
            prior += 0.2
        if same_color_opp >= 2:
            prior += 0.25
        
        cash_after = state.cash[player] - prop.price
        if cash_after < 50:
            prior -= 0.2
        elif cash_after > 200:
            prior += 0.1
        
        return max(0.1, min(0.9, prior))
    
    def _simulate_turn_completion(self, state: GameState, player: int) -> GameState:
        """Complete current turn and switch to opponent"""
        state.current_player = 1 - player
        state.turn_count += 1
        state.check_game_over()
        return state
    
    def _simulate(self, state: GameState, starting_player: int) -> float:
        """
        Enhanced simulation with smart policy
        """
        sim_state = state.copy()
        depth = 0
        
        while not sim_state.game_over and depth < self.max_simulation_depth:
            current_player = sim_state.current_player
            
            # Roll dice and move
            dice_roll = sim_state.roll_dice()
            old_pos = sim_state.positions[current_player]
            new_pos = sim_state.move_player(current_player, dice_roll)
            
            # Handle landing
            if sim_state.is_gamble_tile(new_pos):
                effect = random.choice(GAMBLE_EFFECTS)
                sim_state.apply_gamble_effect(current_player, effect)
            else:
                prop = sim_state.get_property_at(new_pos)
                if prop:
                    if prop.owner is not None and prop.owner != current_player:
                        sim_state.pay_fare(current_player, prop)
                    elif prop.owner is None:
                        should_buy = self._smart_simulation_policy(sim_state, current_player, prop)
                        if should_buy:
                            sim_state.buy_property(current_player, prop)
            
            sim_state.current_player = 1 - current_player
            sim_state.turn_count += 1
            sim_state.check_game_over()
            depth += 1
        
        return self._calculate_reward(sim_state)
    
    def _smart_simulation_policy(self, state: GameState, player: int, prop) -> bool:
        """
        Advanced simulation policy that mimics strategic play
        """
        if state.cash[player] < prop.price:
            return False
        
        cash_after = state.cash[player] - prop.price
        opponent = 1 - player
        
        player_props = state.get_player_properties(player)
        opp_props = state.get_player_properties(opponent)
        
        same_color_mine = [p for p in player_props if p.color == prop.color]
        same_color_opp = [p for p in opp_props if p.color == prop.color]
        total_same_color = len([p for p in state.properties if p.color == prop.color])
        
        # ALWAYS complete monopoly
        if len(same_color_mine) == total_same_color - 1:
            return True
        
        # ALWAYS block opponent monopoly
        if len(same_color_opp) == total_same_color - 1:
            if cash_after >= 20:
                return True
        
        # Strong block
        if len(same_color_opp) >= total_same_color - 2 and len(same_color_mine) == 0:
            if cash_after >= 40:
                return True
        
        # Building towards monopoly
        if len(same_color_mine) >= 1 and len(same_color_opp) == 0:
            if cash_after >= 60:
                return True
            return random.random() < 0.7
        
        # Calculate dynamic risk reserve
        max_opp_fare = max([p.fare for p in opp_props], default=0)
        opp_monopolies = self._count_monopolies(state, opponent)
        
        # If opponent has monopoly, need more reserve
        if opp_monopolies > 0:
            safe_reserve = max(150, max_opp_fare * 3)
        else:
            safe_reserve = max(80, max_opp_fare * 2)
        
        unsold = len(state.get_unowned_properties())
        
        # Phase-based buying
        if unsold > 28:  # Very early
            if cash_after >= 50:
                return True
            return random.random() < 0.8
        
        if unsold > 22:  # Early
            if cash_after >= 80:
                return True
            return random.random() < 0.7
        
        if unsold > 15:  # Mid
            if cash_after >= safe_reserve:
                return True
            return random.random() < 0.5
        
        if unsold > 8:  # Late-mid
            if cash_after >= safe_reserve * 1.2:
                return True
            return random.random() < 0.35
        
        # Late game - very conservative
        if cash_after >= safe_reserve * 1.5:
            return True
        return random.random() < 0.2
    
    def _count_monopolies(self, state: GameState, player: int) -> int:
        """Count how many monopolies a player has"""
        props = state.get_player_properties(player)
        colors = {}
        for p in props:
            colors[p.color] = colors.get(p.color, 0) + 1
        
        color_totals = {}
        for p in state.properties:
            color_totals[p.color] = color_totals.get(p.color, 0) + 1
        
        monopolies = 0
        for color, count in colors.items():
            if count == color_totals.get(color, 0):
                monopolies += 1
        
        return monopolies
    
    def _calculate_reward(self, state: GameState) -> float:
        """
        Enhanced reward calculation with monopoly awareness
        """
        if state.game_over:
            if state.winner == self.player_id:
                return 1.0
            elif state.winner == 1 - self.player_id:
                return -1.0
            else:
                return 0.0  # Draw
        
        # Comprehensive evaluation
        my_cash = state.cash[self.player_id]
        opp_cash = state.cash[1 - self.player_id]
        
        my_props = state.get_player_properties(self.player_id)
        opp_props = state.get_player_properties(1 - self.player_id)
        
        my_value = sum(p.price for p in my_props)
        opp_value = sum(p.price for p in opp_props)
        
        my_fare_potential = sum(p.fare for p in my_props)
        opp_fare_potential = sum(p.fare for p in opp_props)
        
        # Check monopolies
        my_monopolies = self._count_monopolies(state, self.player_id)
        opp_monopolies = self._count_monopolies(state, 1 - self.player_id)
        
        # Calculate score components
        wealth_diff = (my_cash + my_value) - (opp_cash + opp_value)
        fare_diff = my_fare_potential - opp_fare_potential
        monopoly_diff = my_monopolies - opp_monopolies
        
        # Weighted combination
        score = (
            wealth_diff * 0.3 +
            fare_diff * 4.0 +  # Fare is very important
            monopoly_diff * 300 +  # Monopolies are crucial
            len(my_props) * 15 -
            len(opp_props) * 15
        )
        
        # Normalize to [-1, 1]
        max_score = 3000
        normalized = score / max_score
        
        return max(-1.0, min(1.0, normalized))
    
    def _backpropagate(self, node: MCTSNode, reward: float) -> None:
        """
        Backpropagation with variance tracking
        """
        current = node
        
        while current is not None:
            current.visits += 1
            
            if current.player == self.player_id:
                current.total_reward += reward
                current.squared_reward += reward ** 2
            else:
                current.total_reward -= reward
                current.squared_reward += reward ** 2
            
            current = current.parent
    
    def _update_action_stats(self, action: str, state: GameState):
        """Track action outcomes for learning"""
        if action in self.action_stats:
            self.action_stats[action]["games"] += 1
    
    def get_stats(self) -> dict:
        return {
            "name": self.get_name(),
            "player_id": self.player_id,
            "num_simulations": self.num_simulations,
            "total_simulations_run": self.total_simulations,
            "exploration_constant": self.exploration_constant,
            "algorithm": "Enhanced MCTS with UCB1-Tuned and Strategic Priors"
        }


class HybridMCTSAgent(MCTSAgent):
    """
    Hybrid MCTS with deeper search and stronger heuristics
    """
    
    def __init__(self, player_id: int, num_simulations: int = 1500,
                 simulation_depth: int = 40):
        super().__init__(player_id, num_simulations, 1.0, simulation_depth)
        self.simulation_depth = simulation_depth
    
    def get_name(self) -> str:
        return f"Hybrid MCTS (sims={self.num_simulations}, depth={self.simulation_depth})"
    
    def _simulate(self, state: GameState, starting_player: int) -> float:
        """
        Shorter but smarter simulation with early cutoff
        """
        sim_state = state.copy()
        
        for depth in range(self.simulation_depth):
            if sim_state.game_over:
                break
            
            current_player = sim_state.current_player
            
            dice_roll = sim_state.roll_dice()
            new_pos = sim_state.move_player(current_player, dice_roll)
            
            if sim_state.is_gamble_tile(new_pos):
                effect = random.choice(GAMBLE_EFFECTS)
                sim_state.apply_gamble_effect(current_player, effect)
            else:
                prop = sim_state.get_property_at(new_pos)
                if prop:
                    if prop.owner is not None and prop.owner != current_player:
                        sim_state.pay_fare(current_player, prop)
                    elif prop.owner is None:
                        should_buy = self._smart_simulation_policy(sim_state, current_player, prop)
                        if should_buy:
                            sim_state.buy_property(current_player, prop)
            
            sim_state.current_player = 1 - current_player
            sim_state.turn_count += 1
            sim_state.check_game_over()
            
            # Early termination if clear winner emerging
            if depth > 20:
                my_wealth = sim_state.cash[self.player_id] + sum(p.price for p in sim_state.get_player_properties(self.player_id))
                opp_wealth = sim_state.cash[1 - self.player_id] + sum(p.price for p in sim_state.get_player_properties(1 - self.player_id))
                if abs(my_wealth - opp_wealth) > 1000:
                    break
        
        return self._calculate_reward(sim_state)
    
    def get_stats(self) -> dict:
        stats = super().get_stats()
        stats["algorithm"] = "Hybrid MCTS with Smart Cutoff"
        return stats
