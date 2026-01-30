"""
Game State Management for AI Monopoly
Handles the core game logic, board configuration, and state representation
"""

import copy
import random
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from enum import Enum

class TileType(Enum):
    PROPERTY = "property"
    GAMBLE = "gamble"
    START = "start"

@dataclass
class Property:
    index: int
    name: str
    color: str
    price: int
    fare: int
    owner: Optional[int] = None  # None = unowned, 0 = Agent1, 1 = Agent2

@dataclass
class GambleTile:
    index: int
    name: str

@dataclass
class GambleEffect:
    name: str
    description: str
    effect_type: str  # "cash_change", "opponent_pay", "teleport", "move"
    value: int
    probability: float = 1.0

# Predefined gambling effects
GAMBLE_EFFECTS = [
    GambleEffect("Jackpot!", "Win $300", "cash_change", 300),
    GambleEffect("Lucky Day", "Win $200", "cash_change", 200),
    GambleEffect("Small Win", "Win $100", "cash_change", 100),
    GambleEffect("Tax Refund", "Win $150", "cash_change", 150),
    GambleEffect("Bad Luck", "Lose $100", "cash_change", -100),
    GambleEffect("Unlucky", "Lose $150", "cash_change", -150),
    GambleEffect("Disaster", "Lose $200", "cash_change", -200),
    GambleEffect("Pay Opponent", "Pay opponent $100", "opponent_pay", -100),
    GambleEffect("Receive from Opponent", "Receive $100 from opponent", "opponent_pay", 100),
    GambleEffect("Nothing", "Nothing happens", "cash_change", 0),
]

# Color configuration for properties
COLORS = [
    ("Brown", 60, 30),      # (color, base_price, base_fare)
    ("Light Blue", 80, 40),
    ("Pink", 100, 50),
    ("Orange", 120, 60),
    ("Red", 150, 75),
    ("Yellow", 180, 90),
    ("Green", 220, 110),
    ("Dark Blue", 280, 140),
    ("Purple", 320, 160),
]

# Gamble tile positions (4 positions out of 40)
GAMBLE_POSITIONS = [9, 19, 29, 39]

def create_board() -> Tuple[List[dict], List[Property], List[GambleTile]]:
    """Create the game board with 36 properties and 4 gamble tiles"""
    board = []
    properties = []
    gamble_tiles = []
    
    property_idx = 0
    
    for i in range(40):
        if i in GAMBLE_POSITIONS:
            tile = {
                "index": i,
                "type": TileType.GAMBLE.value,
                "name": f"Gamble Zone {len(gamble_tiles) + 1}"
            }
            gamble_tiles.append(GambleTile(i, tile["name"]))
        else:
            color_idx = property_idx // 4
            color_name, base_price, base_fare = COLORS[color_idx]
            position_in_color = property_idx % 4
            
            # Slightly vary prices within same color
            price = base_price + (position_in_color * 10)
            fare = base_fare + (position_in_color * 5)
            
            prop = Property(
                index=i,
                name=f"{color_name} Property {position_in_color + 1}",
                color=color_name,
                price=price,
                fare=fare
            )
            properties.append(prop)
            
            tile = {
                "index": i,
                "type": TileType.PROPERTY.value,
                "name": prop.name,
                "color": prop.color,
                "price": prop.price,
                "fare": prop.fare,
                "owner": None
            }
            property_idx += 1
        
        board.append(tile)
    
    return board, properties, gamble_tiles

@dataclass
class GameState:
    """Complete game state representation"""
    positions: List[int] = field(default_factory=lambda: [0, 0])  # [agent1_pos, agent2_pos]
    cash: List[int] = field(default_factory=lambda: [1500, 1500])  # Starting cash
    properties: List[Property] = field(default_factory=list)
    board: List[dict] = field(default_factory=list)
    gamble_tiles: List[GambleTile] = field(default_factory=list)
    current_player: int = 0  # 0 or 1
    turn_count: int = 0
    max_turns: int = 200
    game_over: bool = False
    winner: Optional[int] = None
    last_dice_roll: int = 0
    last_action: str = ""
    last_gamble_effect: Optional[str] = None
    
    def __post_init__(self):
        if not self.board:
            self.board, self.properties, self.gamble_tiles = create_board()
    
    def copy(self) -> 'GameState':
        """Create a deep copy of the game state"""
        new_state = GameState(
            positions=self.positions.copy(),
            cash=self.cash.copy(),
            properties=[copy.deepcopy(p) for p in self.properties],
            board=copy.deepcopy(self.board),
            gamble_tiles=self.gamble_tiles.copy(),
            current_player=self.current_player,
            turn_count=self.turn_count,
            max_turns=self.max_turns,
            game_over=self.game_over,
            winner=self.winner,
            last_dice_roll=self.last_dice_roll,
            last_action=self.last_action,
            last_gamble_effect=self.last_gamble_effect
        )
        return new_state
    
    def roll_dice(self) -> int:
        """Roll two dice (2-12)"""
        return random.randint(1, 6) + random.randint(1, 6)
    
    def get_property_at(self, position: int) -> Optional[Property]:
        """Get the property at a given position"""
        for prop in self.properties:
            if prop.index == position:
                return prop
        return None
    
    def is_gamble_tile(self, position: int) -> bool:
        """Check if position is a gamble tile"""
        return position in GAMBLE_POSITIONS
    
    def get_unowned_properties(self) -> List[Property]:
        """Get list of unowned properties"""
        return [p for p in self.properties if p.owner is None]
    
    def all_properties_sold(self) -> bool:
        """Check if all properties are sold"""
        return len(self.get_unowned_properties()) == 0
    
    def get_player_properties(self, player: int) -> List[Property]:
        """Get properties owned by a player"""
        return [p for p in self.properties if p.owner == player]
    
    def get_player_property_value(self, player: int) -> int:
        """Get total property value for a player"""
        return sum(p.price for p in self.get_player_properties(player))
    
    def get_player_fare_income(self, player: int) -> int:
        """Get total potential fare income for a player"""
        return sum(p.fare for p in self.get_player_properties(player))
    
    def apply_gamble_effect(self, player: int, effect: GambleEffect) -> str:
        """Apply a gamble effect to the player"""
        opponent = 1 - player
        
        if effect.effect_type == "cash_change":
            self.cash[player] += effect.value
            self.cash[player] = max(0, self.cash[player])
        elif effect.effect_type == "opponent_pay":
            if effect.value > 0:
                # Receive from opponent
                amount = min(effect.value, self.cash[opponent])
                self.cash[opponent] -= amount
                self.cash[player] += amount
            else:
                # Pay to opponent
                amount = min(-effect.value, self.cash[player])
                self.cash[player] -= amount
                self.cash[opponent] += amount
        
        return effect.description
    
    def move_player(self, player: int, dice_roll: int) -> int:
        """Move a player by dice roll amount"""
        old_pos = self.positions[player]
        new_pos = (old_pos + dice_roll) % 40
        self.positions[player] = new_pos
        return new_pos
    
    def can_buy_property(self, player: int, prop: Property) -> bool:
        """Check if player can buy a property"""
        return prop.owner is None and self.cash[player] >= prop.price
    
    def buy_property(self, player: int, prop: Property) -> bool:
        """Player buys a property"""
        if self.can_buy_property(player, prop):
            self.cash[player] -= prop.price
            prop.owner = player
            # Update board
            for tile in self.board:
                if tile["index"] == prop.index:
                    tile["owner"] = player
            return True
        return False
    
    def pay_fare(self, player: int, prop: Property) -> int:
        """Player pays fare to property owner"""
        if prop.owner is not None and prop.owner != player:
            fare = prop.fare
            # Check for color bonus (own all properties of same color)
            owner_props = self.get_player_properties(prop.owner)
            same_color = [p for p in owner_props if p.color == prop.color]
            color_count = len([p for p in self.properties if p.color == prop.color])
            
            if len(same_color) == color_count:
                fare *= 2  # Double fare for monopoly
            
            actual_payment = min(fare, self.cash[player])
            self.cash[player] -= actual_payment
            self.cash[prop.owner] += actual_payment
            return actual_payment
        return 0
    
    def check_game_over(self) -> bool:
        """Check if game is over"""
        # Game over conditions:
        # 1. All properties sold AND one player passes start
        # 2. Max turns reached
        # 3. A player goes bankrupt
        
        if self.cash[0] <= 0 or self.cash[1] <= 0:
            self.game_over = True
            self.winner = 0 if self.cash[0] > self.cash[1] else 1
            return True
        
        if self.turn_count >= self.max_turns:
            self.game_over = True
            self.winner = self._determine_winner()
            return True
        
        if self.all_properties_sold() and self.turn_count > 40:
            self.game_over = True
            self.winner = self._determine_winner()
            return True
        
        return False
    
    def _determine_winner(self) -> int:
        """Determine winner based on total wealth"""
        total_0 = self.cash[0] + self.get_player_property_value(0)
        total_1 = self.cash[1] + self.get_player_property_value(1)
        
        if total_0 > total_1:
            return 0
        elif total_1 > total_0:
            return 1
        else:
            return 0  # Tie goes to player 0
    
    def get_available_actions(self, player: int) -> List[str]:
        """Get available actions for current state"""
        pos = self.positions[player]
        prop = self.get_property_at(pos)
        
        if prop and prop.owner is None and self.cash[player] >= prop.price:
            return ["BUY", "SKIP"]
        return ["SKIP"]  # No decision needed
    
    def evaluate(self, player: int) -> float:
        """Evaluate state from player's perspective"""
        opponent = 1 - player
        
        # Cash difference
        cash_diff = self.cash[player] - self.cash[opponent]
        
        # Property value difference
        prop_value_diff = self.get_player_property_value(player) - self.get_player_property_value(opponent)
        
        # Fare income potential
        fare_diff = self.get_player_fare_income(player) - self.get_player_fare_income(opponent)
        
        # Count monopolies (all properties of same color)
        my_monopolies = self._count_monopolies(player)
        opp_monopolies = self._count_monopolies(opponent)
        monopoly_bonus = (my_monopolies - opp_monopolies) * 200
        
        # Weights
        return cash_diff + 0.5 * prop_value_diff + 0.3 * fare_diff + monopoly_bonus
    
    def _count_monopolies(self, player: int) -> int:
        """Count how many complete color sets a player owns"""
        player_props = self.get_player_properties(player)
        color_counts = {}
        
        for prop in player_props:
            color_counts[prop.color] = color_counts.get(prop.color, 0) + 1
        
        monopolies = 0
        for color, count in color_counts.items():
            total_in_color = len([p for p in self.properties if p.color == color])
            if count == total_in_color:
                monopolies += 1
        
        return monopolies
    
    def to_dict(self) -> dict:
        """Convert game state to dictionary for JSON serialization"""
        return {
            "positions": self.positions,
            "cash": self.cash,
            "currentPlayer": self.current_player,
            "turnCount": self.turn_count,
            "gameOver": self.game_over,
            "winner": self.winner,
            "lastDiceRoll": self.last_dice_roll,
            "lastAction": self.last_action,
            "lastGambleEffect": self.last_gamble_effect,
            "board": self.board,
            "properties": [
                {
                    "index": p.index,
                    "name": p.name,
                    "color": p.color,
                    "price": p.price,
                    "fare": p.fare,
                    "owner": p.owner
                }
                for p in self.properties
            ],
            "playerStats": [
                {
                    "player": i,
                    "cash": self.cash[i],
                    "position": self.positions[i],
                    "propertyCount": len(self.get_player_properties(i)),
                    "propertyValue": self.get_player_property_value(i),
                    "fareIncome": self.get_player_fare_income(i)
                }
                for i in range(2)
            ]
        }
