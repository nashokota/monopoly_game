"""
Monte Carlo Tree Search (MCTS) Agent for AI Monopoly
Uses simulation-based approach to handle uncertainty in dice and gambling
"""

import random
import math
from typing import Optional, List, Dict
from dataclasses import dataclass, field
from game_state import GameState, GAMBLE_EFFECTS

@dataclass
class MCTSNode:
    """Node in the MCTS tree"""
    state: GameState
    parent: Optional['MCTSNode'] = None
    action: Optional[str] = None  # Action that led to this node
    children: List['MCTSNode'] = field(default_factory=list)
    visits: int = 0
    total_reward: float = 0.0
    untried_actions: List[str] = field(default_factory=list)
    player: int = 0  # Player who made the move to get here
    
    def ucb1(self, exploration_constant: float = 1.414) -> float:
        """Calculate UCB1 score for node selection"""
        if self.visits == 0:
            return float('inf')
        
        exploitation = self.total_reward / self.visits
        exploration = exploration_constant * math.sqrt(math.log(self.parent.visits) / self.visits)
        
        return exploitation + exploration
    
    def best_child(self, exploration_constant: float = 1.414) -> 'MCTSNode':
        """Select best child based on UCB1"""
        return max(self.children, key=lambda c: c.ucb1(exploration_constant))
    
    def is_fully_expanded(self) -> bool:
        """Check if all actions have been tried"""
        return len(self.untried_actions) == 0
    
    def is_terminal(self) -> bool:
        """Check if this is a terminal state"""
        return self.state.game_over


class MCTSAgent:
    """
    Monte Carlo Tree Search Agent
    
    MCTS naturally handles stochastic elements through random simulations.
    Perfect for games with dice rolls and random events.
    
    Four phases:
    1. Selection: Use UCB1 to select promising nodes
    2. Expansion: Add new child node for unexplored action
    3. Simulation: Random playout until terminal state
    4. Backpropagation: Update node statistics
    """
    
    def __init__(self, player_id: int, num_simulations: int = 1000, 
                 exploration_constant: float = 1.414, max_simulation_depth: int = 50):
        self.player_id = player_id
        self.num_simulations = num_simulations
        self.exploration_constant = exploration_constant
        self.max_simulation_depth = max_simulation_depth
        self.total_simulations = 0
    
    def get_name(self) -> str:
        return f"MCTS (sims={self.num_simulations})"
    
    def choose_action(self, state: GameState) -> str:
        """
        Choose the best action using MCTS.
        """
        actions = state.get_available_actions(self.player_id)
        
        if len(actions) == 1:
            return actions[0]
        
        # Create root node
        root = MCTSNode(
            state=state.copy(),
            untried_actions=actions.copy(),
            player=self.player_id
        )
        
        # Run simulations
        for _ in range(self.num_simulations):
            self.total_simulations += 1
            
            # Selection & Expansion
            node = self._select(root)
            
            # Simulation
            reward = self._simulate(node.state, node.player)
            
            # Backpropagation
            self._backpropagate(node, reward)
        
        # Choose best action (most visited child)
        if not root.children:
            return random.choice(actions)
        
        best_child = max(root.children, key=lambda c: c.visits)
        return best_child.action
    
    def _select(self, node: MCTSNode) -> MCTSNode:
        """
        Selection phase: traverse tree using UCB1 until we find
        a node that is not fully expanded or is terminal.
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
        Expansion phase: add a new child node for an untried action.
        """
        if not node.untried_actions:
            return node
        
        action = node.untried_actions.pop()
        
        # Create new state by applying action
        new_state = node.state.copy()
        pos = new_state.positions[node.player]
        prop = new_state.get_property_at(pos)
        
        if action == "BUY" and prop and new_state.cash[node.player] >= prop.price:
            new_state.buy_property(node.player, prop)
        
        # Simulate the rest of this turn and opponent's response
        new_state = self._simulate_turn_completion(new_state, node.player)
        
        # Determine next player's available actions
        next_player = new_state.current_player
        next_actions = new_state.get_available_actions(next_player)
        
        child = MCTSNode(
            state=new_state,
            parent=node,
            action=action,
            untried_actions=next_actions,
            player=next_player
        )
        
        node.children.append(child)
        return child
    
    def _simulate_turn_completion(self, state: GameState, player: int) -> GameState:
        """
        Complete the current turn after a decision is made.
        Switch to opponent and prepare for their turn.
        """
        state.current_player = 1 - player
        state.turn_count += 1
        state.check_game_over()
        return state
    
    def _simulate(self, state: GameState, starting_player: int) -> float:
        """
        Simulation phase: random playout until terminal state or depth limit.
        Returns reward from the perspective of self.player_id.
        """
        sim_state = state.copy()
        depth = 0
        
        while not sim_state.game_over and depth < self.max_simulation_depth:
            current_player = sim_state.current_player
            
            # Roll dice and move
            dice_roll = sim_state.roll_dice()
            new_pos = sim_state.move_player(current_player, dice_roll)
            
            # Handle landing
            if sim_state.is_gamble_tile(new_pos):
                # Random gambling effect
                effect = random.choice(GAMBLE_EFFECTS)
                sim_state.apply_gamble_effect(current_player, effect)
            else:
                prop = sim_state.get_property_at(new_pos)
                if prop:
                    if prop.owner is not None and prop.owner != current_player:
                        sim_state.pay_fare(current_player, prop)
                    elif prop.owner is None:
                        # Random buy decision with some heuristic
                        should_buy = self._simulation_policy(sim_state, current_player, prop)
                        if should_buy:
                            sim_state.buy_property(current_player, prop)
            
            # Switch player
            sim_state.current_player = 1 - current_player
            sim_state.turn_count += 1
            sim_state.check_game_over()
            depth += 1
        
        # Calculate reward
        return self._calculate_reward(sim_state)
    
    def _simulation_policy(self, state: GameState, player: int, prop) -> bool:
        """
        Smart policy for simulation with strategic buying decisions.
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
        
        # Always buy to complete monopoly
        if len(same_color_mine) == total_same_color - 1:
            return True
        
        # Always buy to block opponent's monopoly
        if len(same_color_opp) >= total_same_color - 2:
            if cash_after >= 30:
                return True
        
        # Calculate risk-adjusted reserve
        max_opp_fare = max([p.fare for p in opp_props], default=0) * 2
        safe_reserve = max(80, max_opp_fare)
        
        # Early game: aggressive buying
        unsold = len(state.get_unowned_properties())
        if unsold > 25:
            if cash_after >= 30:
                return True
            return random.random() < 0.6
        
        # Mid game: balanced
        if unsold > 15:
            if cash_after >= safe_reserve:
                return True
            return random.random() < 0.4
        
        # Late game: conservative
        if cash_after >= safe_reserve * 1.5:
            return True
        return random.random() < 0.2
    
    def _calculate_reward(self, state: GameState) -> float:
        """
        Calculate reward from self.player_id's perspective.
        Normalized to [-1, 1] range.
        """
        if state.game_over:
            if state.winner == self.player_id:
                return 1.0
            else:
                return -1.0
        
        # Use evaluation function for non-terminal states
        eval_score = state.evaluate(self.player_id)
        
        # Normalize (rough normalization based on typical game values)
        max_score = 5000  # Approximate max score difference
        normalized = eval_score / max_score
        
        return max(-1.0, min(1.0, normalized))
    
    def _backpropagate(self, node: MCTSNode, reward: float) -> None:
        """
        Backpropagation phase: update statistics up the tree.
        """
        current = node
        
        while current is not None:
            current.visits += 1
            
            # Adjust reward based on whose perspective
            if current.player == self.player_id:
                current.total_reward += reward
            else:
                current.total_reward -= reward  # Opponent wants opposite
            
            current = current.parent
    
    def get_stats(self) -> dict:
        """Return agent statistics"""
        return {
            "name": self.get_name(),
            "player_id": self.player_id,
            "num_simulations": self.num_simulations,
            "total_simulations_run": self.total_simulations,
            "exploration_constant": self.exploration_constant,
            "algorithm": "Monte Carlo Tree Search (UCB1)"
        }


class HybridMCTSAgent(MCTSAgent):
    """
    Hybrid MCTS that uses heuristic evaluation in simulations
    instead of playing to terminal state.
    """
    
    def __init__(self, player_id: int, num_simulations: int = 500,
                 simulation_depth: int = 20):
        super().__init__(player_id, num_simulations, 1.414, simulation_depth)
        self.simulation_depth = simulation_depth
    
    def get_name(self) -> str:
        return f"Hybrid MCTS (sims={self.num_simulations}, depth={self.simulation_depth})"
    
    def _simulate(self, state: GameState, starting_player: int) -> float:
        """
        Shorter simulation with heuristic evaluation at cutoff.
        """
        sim_state = state.copy()
        
        for _ in range(self.simulation_depth):
            if sim_state.game_over:
                break
            
            current_player = sim_state.current_player
            
            # Roll and move
            dice_roll = sim_state.roll_dice()
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
                        should_buy = self._simulation_policy(sim_state, current_player, prop)
                        if should_buy:
                            sim_state.buy_property(current_player, prop)
            
            sim_state.current_player = 1 - current_player
            sim_state.turn_count += 1
            sim_state.check_game_over()
        
        return self._calculate_reward(sim_state)
    
    def get_stats(self) -> dict:
        stats = super().get_stats()
        stats["algorithm"] = "Hybrid MCTS with Heuristic Cutoff"
        return stats
