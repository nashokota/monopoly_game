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
    buildings: int = 0  # 0-4 buildings per property

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

# Color configuration for properties - Dhaka City Areas
# Each color group represents a district/zone with neighborhoods
# Brown has 3 properties (position 0 is GO), others have 4
COLORS = [
    ("Brown", 60, 30, ["Hazaribagh", "Rayerbazar", "Shyamoli"]),  # 3 properties (GO is at position 0)
    ("Light Blue", 80, 40, ["Mohammadpur", "Adabor", "Lalmatia", "Dhanmondi"]),
    ("Pink", 100, 50, ["Mirpur", "Pallabi", "Kazipara", "Shewrapara"]),
    ("Orange", 120, 60, ["Uttara", "Abdullahpur", "Diabari", "Azampur"]),
    ("Red", 150, 75, ["Banani", "Gulshan", "Baridhara", "Niketan"]),
    ("Yellow", 180, 90, ["Tejgaon", "Farmgate", "Karwan Bazar", "Bijoy Sarani"]),
    ("Green", 220, 110, ["Motijheel", "Paltan", "Kakrail", "Ramna"]),
    ("Dark Blue", 280, 140, ["Old Dhaka", "Lalbagh", "Chawkbazar", "Sadarghat"]),
    ("Purple", 320, 160, ["Bashundhara", "Aftabnagar", "Badda", "Rampura"]),
]

# Gamble tile positions (4 positions out of 40) - Special Dhaka locations
GAMBLE_POSITIONS = [9, 19, 29, 39]
GAMBLE_NAMES = ["Hatirjheel Lake", "Ahsan Manzil", "National Parliament", "Dhaka University"]

def create_board() -> Tuple[List[dict], List[Property], List[GambleTile]]:
    """Create the game board with GO tile, 35 Dhaka properties and 4 landmark gamble tiles"""
    board = []
    properties = []
    gamble_tiles = []
    
    property_idx = 0
    gamble_idx = 0
    
    for i in range(40):
        if i == 0:
            # Position 0 is the GO (Start) tile
            tile = {
                "index": i,
                "type": TileType.START.value,
                "name": "GO"
            }
        elif i in GAMBLE_POSITIONS:
            gamble_name = GAMBLE_NAMES[gamble_idx]
            tile = {
                "index": i,
                "type": TileType.GAMBLE.value,
                "name": gamble_name
            }
            gamble_tiles.append(GambleTile(i, gamble_name))
            gamble_idx += 1
        else:
            # Calculate color index accounting for Brown having only 3 properties
            if property_idx < 3:
                color_idx = 0  # Brown (3 properties)
                position_in_color = property_idx
            else:
                # After Brown, each color has 4 properties
                adjusted_idx = property_idx - 3
                color_idx = 1 + (adjusted_idx // 4)
                position_in_color = adjusted_idx % 4
            
            color_name, base_price, base_fare, area_names = COLORS[color_idx]
            
            # Get the specific Dhaka area name
            area_name = area_names[position_in_color]
            
            # Slightly vary prices within same color
            price = base_price + (position_in_color * 10)
            fare = base_fare + (position_in_color * 5)
            
            prop = Property(
                index=i,
                name=area_name,
                color=color_name,
                price=price,
                fare=fare
            )
            properties.append(prop)
            
            tile = {
                "index": i,
                "type": TileType.PROPERTY.value,
                "name": area_name,
                "color": prop.color,
                "price": prop.price,
                "fare": prop.fare,
                "owner": None,
                "buildings": 0
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
        """Move a player by dice roll amount. Awards $400 for passing GO."""
        old_pos = self.positions[player]
        new_pos = (old_pos + dice_roll) % 40
        self.positions[player] = new_pos
        
        # Check if player passed GO (crossed position 0)
        if new_pos < old_pos:  # Wrapped around the board
            self.cash[player] += 400  # Bonus for passing GO
        
        return new_pos
    
    def has_monopoly(self, player: int, color: str) -> bool:
        """Check if player owns all properties of a color"""
        player_props = self.get_player_properties(player)
        same_color = [p for p in player_props if p.color == color]
        total_in_color = len([p for p in self.properties if p.color == color])
        return len(same_color) == total_in_color
    
    def get_building_cost(self, prop: Property) -> int:
        """Get cost to build on a property (110% of property price)"""
        return int(prop.price * 1.1)
    
    def can_build(self, player: int, prop: Property) -> bool:
        """Check if player can build on a property"""
        if prop.owner != player:
            return False
        if prop.buildings >= 4:  # Max 4 buildings
            return False
        if not self.has_monopoly(player, prop.color):
            return False
        build_cost = self.get_building_cost(prop)
        if self.cash[player] < build_cost:
            return False
        return True
    
    def build_on_property(self, player: int, prop: Property) -> bool:
        """Build one building on a property"""
        if not self.can_build(player, prop):
            return False
        
        build_cost = self.get_building_cost(prop)
        self.cash[player] -= build_cost
        prop.buildings += 1
        
        # Update board
        for tile in self.board:
            if tile["index"] == prop.index:
                tile["buildings"] = prop.buildings
        
        return True
    
    def get_buildable_properties(self, player: int) -> List[Property]:
        """Get list of properties where player can build"""
        return [p for p in self.properties if self.can_build(player, p)]
    
    def get_player_total_buildings(self, player: int) -> int:
        """Get total number of buildings owned by player"""
        return sum(p.buildings for p in self.get_player_properties(player))
    
    def get_sell_building_value(self, prop: Property) -> int:
        """Get value when selling a building (95% of build cost)"""
        build_cost = self.get_building_cost(prop)
        return int(build_cost * 0.95)
    
    def get_sell_property_value(self, prop: Property) -> int:
        """Get value when selling a property (95% of purchase price)"""
        return int(prop.price * 0.95)
    
    def can_sell_building(self, player: int, prop: Property) -> bool:
        """Check if player can sell a building on a property"""
        return prop.owner == player and prop.buildings > 0
    
    def sell_building(self, player: int, prop: Property) -> int:
        """Sell one building on a property. Returns money gained."""
        if not self.can_sell_building(player, prop):
            return 0
        
        sell_value = self.get_sell_building_value(prop)
        self.cash[player] += sell_value
        prop.buildings -= 1
        
        # Update board
        for tile in self.board:
            if tile["index"] == prop.index:
                tile["buildings"] = prop.buildings
        
        return sell_value
    
    def can_sell_property(self, player: int, prop: Property) -> bool:
        """Check if player can sell a property (must have no buildings)"""
        return prop.owner == player and prop.buildings == 0
    
    def sell_property(self, player: int, prop: Property) -> int:
        """Sell a property back to the bank. Returns money gained."""
        if not self.can_sell_property(player, prop):
            return 0
        
        sell_value = self.get_sell_property_value(prop)
        self.cash[player] += sell_value
        prop.owner = None
        
        # Update board
        for tile in self.board:
            if tile["index"] == prop.index:
                tile["owner"] = None
        
        return sell_value
    
    def get_sellable_buildings(self, player: int) -> List[Property]:
        """Get list of properties with buildings that can be sold"""
        return [p for p in self.properties if self.can_sell_building(player, p)]
    
    def get_sellable_properties(self, player: int) -> List[Property]:
        """Get list of properties that can be sold (no buildings)"""
        return [p for p in self.properties if self.can_sell_property(player, p)]
    
    def get_total_sellable_value(self, player: int) -> int:
        """Get total value of all sellable assets"""
        total = 0
        for prop in self.get_player_properties(player):
            # Building value
            total += prop.buildings * self.get_sell_building_value(prop)
            # Property value (only if no buildings or after selling buildings)
            total += self.get_sell_property_value(prop)
        return total
    
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
    
    def calculate_fare(self, player: int, prop: Property) -> Tuple[int, dict]:
        """Calculate fare without actually paying. Returns (fare_amount, fare_details)"""
        fare_details = {"base": prop.fare, "monopolyBonus": False, "buildingBonus": 0, "total": 0}
        
        if prop.owner is not None and prop.owner != player:
            fare = prop.fare
            owner_props = self.get_player_properties(prop.owner)
            same_color = [p for p in owner_props if p.color == prop.color]
            color_count = len([p for p in self.properties if p.color == prop.color])
            
            if len(same_color) == color_count:
                fare *= 2
                fare_details["monopolyBonus"] = True
            
            if prop.buildings > 0:
                building_bonus = int(fare * 0.20 * prop.buildings)
                fare_details["buildingBonus"] = building_bonus
                fare += building_bonus
            
            fare_details["total"] = fare
            return fare, fare_details
        return 0, fare_details
    
    def pay_fare(self, player: int, prop: Property) -> Tuple[int, dict]:
        """Player pays fare to property owner. Returns (amount_paid, fare_details)"""
        fare, fare_details = self.calculate_fare(player, prop)
        
        if fare > 0:
            actual_payment = min(fare, self.cash[player])
            self.cash[player] -= actual_payment
            self.cash[prop.owner] += actual_payment
            return actual_payment, fare_details
        return 0, fare_details
    
    def check_game_over(self) -> bool:
        """Check if game is over"""
        # Game over condition: A player goes bankrupt (cash <= 0)
        
        if self.cash[0] <= 0:
            self.game_over = True
            self.winner = 1  # Player 1 wins when Player 0 is bankrupt
            return True
        
        if self.cash[1] <= 0:
            self.game_over = True
            self.winner = 0  # Player 0 wins when Player 1 is bankrupt
            return True
        
        # Safety limit to prevent infinite games
        if self.turn_count >= self.max_turns:
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
        actions = []
        pos = self.positions[player]
        prop = self.get_property_at(pos)
        
        # Can buy unowned property?
        if prop and prop.owner is None and self.cash[player] >= prop.price:
            actions.extend(["BUY", "SKIP"])
        else:
            actions.append("SKIP")
        
        # Can build on any owned property with monopoly?
        buildable = self.get_buildable_properties(player)
        for buildable_prop in buildable:
            actions.append(f"BUILD_{buildable_prop.index}")
        
        return actions
    
    def get_build_actions(self, player: int) -> List[str]:
        """Get available build actions for a player"""
        buildable = self.get_buildable_properties(player)
        return [f"BUILD_{p.index}" for p in buildable]
    
    def get_sell_building_actions(self, player: int) -> List[str]:
        """Get available sell building actions for a player"""
        sellable = self.get_sellable_buildings(player)
        return [f"SELL_BUILDING_{p.index}" for p in sellable]
    
    def get_sell_property_actions(self, player: int) -> List[str]:
        """Get available sell property actions for a player"""
        sellable = self.get_sellable_properties(player)
        return [f"SELL_PROPERTY_{p.index}" for p in sellable]
    
    def evaluate(self, player: int) -> float:
        """Evaluate state from player's perspective"""
        opponent = 1 - player
        
        # Total wealth (cash + property value + building value) - this is what determines the winner
        my_wealth = self.cash[player] + self.get_player_property_value(player) + self.get_player_building_value(player)
        opp_wealth = self.cash[opponent] + self.get_player_property_value(opponent) + self.get_player_building_value(opponent)
        wealth_diff = my_wealth - opp_wealth
        
        # Property count bonus - owning more properties is strategically valuable
        my_props = len(self.get_player_properties(player))
        opp_props = len(self.get_player_properties(opponent))
        property_count_bonus = (my_props - opp_props) * 50
        
        # Fare income potential (future earnings) - includes building bonuses
        fare_diff = self.get_player_fare_income(player) - self.get_player_fare_income(opponent)
        
        # Count monopolies (all properties of same color) - very valuable
        my_monopolies = self._count_monopolies(player)
        opp_monopolies = self._count_monopolies(opponent)
        monopoly_bonus = (my_monopolies - opp_monopolies) * 300
        
        # Building bonus - buildings significantly increase income potential
        my_buildings = self.get_player_total_buildings(player)
        opp_buildings = self.get_player_total_buildings(opponent)
        building_bonus = (my_buildings - opp_buildings) * 75
        
        # Bonus for properties that could complete a monopoly
        near_monopoly_bonus = self._near_monopoly_value(player) - self._near_monopoly_value(opponent)
        
        # Weights: prioritize wealth, then fare income, then strategic position
        return wealth_diff + 0.8 * fare_diff + property_count_bonus + monopoly_bonus + building_bonus + near_monopoly_bonus
    
    def get_player_building_value(self, player: int) -> int:
        """Get total building value for a player (buildings cost 110% of property price)"""
        total = 0
        for prop in self.get_player_properties(player):
            total += int(prop.price * 1.1) * prop.buildings
        return total
    
    def _near_monopoly_value(self, player: int) -> float:
        """Calculate bonus for being close to completing monopolies"""
        player_props = self.get_player_properties(player)
        color_counts = {}
        
        for prop in player_props:
            color_counts[prop.color] = color_counts.get(prop.color, 0) + 1
        
        bonus = 0
        for color, count in color_counts.items():
            total_in_color = len([p for p in self.properties if p.color == color])
            # Bonus for having 3 out of 4 properties in a color
            if count == total_in_color - 1:
                bonus += 100
            # Smaller bonus for having 2 out of 4
            elif count == total_in_color - 2 and total_in_color == 4:
                bonus += 30
        
        return bonus
    
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
                    "owner": p.owner,
                    "buildings": p.buildings,
                    "buildCost": self.get_building_cost(p) if p.owner is not None else None,
                    "canBuild": self.can_build(p.owner, p) if p.owner is not None else False
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
                    "buildingValue": self.get_player_building_value(i),
                    "buildingCount": self.get_player_total_buildings(i),
                    "fareIncome": self.get_player_fare_income(i),
                    "monopolies": self._count_monopolies(i)
                }
                for i in range(2)
            ],
            "buildableProperties": [
                {
                    "index": p.index,
                    "name": p.name,
                    "color": p.color,
                    "buildCost": self.get_building_cost(p),
                    "currentBuildings": p.buildings,
                    "owner": p.owner
                }
                for p in self.get_buildable_properties(0) + self.get_buildable_properties(1)
            ]
        }
