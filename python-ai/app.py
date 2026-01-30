"""
Flask API Server for AI Monopoly
Provides REST endpoints for the React frontend
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import random
from typing import Dict, Optional

from game_state import GameState, GAMBLE_EFFECTS
from minimax_agent import ExpectiminimaxAgent, SimplifiedMinimaxAgent
from mcts_agent import MCTSAgent, HybridMCTSAgent
from game_engine import GameEngine

app = Flask(__name__)
CORS(app)

# Store active games
active_games: Dict[str, dict] = {}

def create_agent(agent_type: str, player_id: int, config: dict = None):
    """Factory function to create agents based on type"""
    config = config or {}
    
    if agent_type == "expectiminimax":
        return ExpectiminimaxAgent(
            player_id=player_id,
            max_depth=config.get("depth", 4),
            use_sampling=config.get("use_sampling", True),
            samples=config.get("samples", 5)
        )
    elif agent_type == "minimax":
        return SimplifiedMinimaxAgent(
            player_id=player_id,
            max_depth=config.get("depth", 6)
        )
    elif agent_type == "mcts":
        return MCTSAgent(
            player_id=player_id,
            num_simulations=config.get("simulations", 500),
            exploration_constant=config.get("exploration", 1.414),
            max_simulation_depth=config.get("max_depth", 50)
        )
    elif agent_type == "hybrid_mcts":
        return HybridMCTSAgent(
            player_id=player_id,
            num_simulations=config.get("simulations", 300),
            simulation_depth=config.get("depth", 20)
        )
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "service": "AI Monopoly Python Backend"})


@app.route('/api/agents', methods=['GET'])
def get_available_agents():
    """Get list of available AI agents"""
    return jsonify({
        "agents": [
            {
                "id": "expectiminimax",
                "name": "Expectiminimax",
                "description": "Minimax with chance nodes for dice rolls. Best for strategic play.",
                "config": {
                    "depth": {"type": "number", "default": 4, "min": 1, "max": 8},
                    "use_sampling": {"type": "boolean", "default": True},
                    "samples": {"type": "number", "default": 5, "min": 1, "max": 20}
                }
            },
            {
                "id": "minimax",
                "name": "Simplified Minimax",
                "description": "Standard minimax treating dice as average. Fast but less accurate.",
                "config": {
                    "depth": {"type": "number", "default": 6, "min": 1, "max": 10}
                }
            },
            {
                "id": "mcts",
                "name": "Monte Carlo Tree Search",
                "description": "Simulation-based search. Great for handling uncertainty.",
                "config": {
                    "simulations": {"type": "number", "default": 500, "min": 100, "max": 2000},
                    "exploration": {"type": "number", "default": 1.414, "min": 0.5, "max": 3.0},
                    "max_depth": {"type": "number", "default": 50, "min": 10, "max": 100}
                }
            },
            {
                "id": "hybrid_mcts",
                "name": "Hybrid MCTS",
                "description": "MCTS with heuristic evaluation cutoff. Balanced speed and quality.",
                "config": {
                    "simulations": {"type": "number", "default": 300, "min": 50, "max": 1000},
                    "depth": {"type": "number", "default": 20, "min": 5, "max": 50}
                }
            }
        ]
    })


@app.route('/api/game/new', methods=['POST'])
def new_game():
    """Start a new game"""
    data = request.json or {}
    
    agent1_type = data.get("agent1", {}).get("type", "expectiminimax")
    agent1_config = data.get("agent1", {}).get("config", {})
    agent2_type = data.get("agent2", {}).get("type", "mcts")
    agent2_config = data.get("agent2", {}).get("config", {})
    
    starting_cash = data.get("startingCash", 1500)
    max_turns = data.get("maxTurns", 200)
    
    try:
        agent1 = create_agent(agent1_type, 0, agent1_config)
        agent2 = create_agent(agent2_type, 1, agent2_config)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    
    # Generate game ID
    game_id = f"game_{random.randint(10000, 99999)}"
    
    # Create game state
    state = GameState(
        cash=[starting_cash, starting_cash],
        max_turns=max_turns
    )
    
    # Store game
    active_games[game_id] = {
        "state": state,
        "agents": [agent1, agent2],
        "engine": GameEngine(agent1, agent2, starting_cash, max_turns),
        "history": []
    }
    
    return jsonify({
        "gameId": game_id,
        "state": state.to_dict(),
        "agents": [
            {"id": 0, "type": agent1_type, "name": agent1.get_name()},
            {"id": 1, "type": agent2_type, "name": agent2.get_name()}
        ]
    })


@app.route('/api/game/<game_id>/state', methods=['GET'])
def get_game_state(game_id: str):
    """Get current game state"""
    if game_id not in active_games:
        return jsonify({"error": "Game not found"}), 404
    
    game = active_games[game_id]
    return jsonify({
        "gameId": game_id,
        "state": game["state"].to_dict(),
        "history": game["history"][-10:]  # Last 10 turns
    })


@app.route('/api/game/<game_id>/turn', methods=['POST'])
def play_turn(game_id: str):
    """Play a single turn"""
    if game_id not in active_games:
        return jsonify({"error": "Game not found"}), 404
    
    game = active_games[game_id]
    state = game["state"]
    
    if state.game_over:
        return jsonify({
            "gameId": game_id,
            "state": state.to_dict(),
            "gameOver": True,
            "winner": state.winner,
            "message": "Game is already over"
        })
    
    # Play one turn
    current_player = state.current_player
    agent = game["agents"][current_player]
    
    turn_info = {
        "turn": state.turn_count,
        "player": current_player,
        "agent": agent.get_name(),
        "initialCash": state.cash.copy(),
        "initialPosition": state.positions[current_player],
    }
    
    # Roll dice
    dice_roll = state.roll_dice()
    state.last_dice_roll = dice_roll
    turn_info["diceRoll"] = dice_roll
    
    # Move player
    new_pos = state.move_player(current_player, dice_roll)
    turn_info["newPosition"] = new_pos
    
    # Handle landing
    if state.is_gamble_tile(new_pos):
        effect = random.choice(GAMBLE_EFFECTS)
        result = state.apply_gamble_effect(current_player, effect)
        state.last_gamble_effect = f"{effect.name}: {result}"
        turn_info["landedOn"] = "Gamble Tile"
        turn_info["gambleEffect"] = {"name": effect.name, "description": result}
    else:
        prop = state.get_property_at(new_pos)
        if prop:
            turn_info["landedOn"] = {
                "type": "property",
                "name": prop.name,
                "color": prop.color,
                "price": prop.price,
                "fare": prop.fare,
                "owner": prop.owner
            }
            
            if prop.owner is not None and prop.owner != current_player:
                fare = state.pay_fare(current_player, prop)
                turn_info["farePaid"] = fare
            elif prop.owner is None:
                action = agent.choose_action(state)
                turn_info["action"] = action
                state.last_action = action
                
                if action == "BUY":
                    state.buy_property(current_player, prop)
                    turn_info["propertyBought"] = prop.name
    
    turn_info["finalCash"] = state.cash.copy()
    
    # Switch player and advance turn
    state.current_player = 1 - current_player
    state.turn_count += 1
    state.check_game_over()
    
    game["history"].append(turn_info)
    
    return jsonify({
        "gameId": game_id,
        "state": state.to_dict(),
        "turnInfo": turn_info,
        "gameOver": state.game_over,
        "winner": state.winner
    })


@app.route('/api/game/<game_id>/play', methods=['POST'])
def play_full_game(game_id: str):
    """Play the entire game to completion"""
    if game_id not in active_games:
        return jsonify({"error": "Game not found"}), 404
    
    game = active_games[game_id]
    engine = game["engine"]
    
    # Reset and play full game
    final_state, winner, history = engine.play_game(verbose=False)
    
    # Update stored game
    game["state"] = final_state
    game["history"] = history
    
    return jsonify({
        "gameId": game_id,
        "state": final_state.to_dict(),
        "winner": winner,
        "totalTurns": final_state.turn_count,
        "history": history
    })


@app.route('/api/game/<game_id>/fast-forward', methods=['POST'])
def fast_forward(game_id: str):
    """Play multiple turns at once"""
    if game_id not in active_games:
        return jsonify({"error": "Game not found"}), 404
    
    data = request.json or {}
    num_turns = min(data.get("turns", 10), 50)  # Max 50 turns at once
    
    game = active_games[game_id]
    state = game["state"]
    
    turns_played = []
    
    for _ in range(num_turns):
        if state.game_over:
            break
        
        current_player = state.current_player
        agent = game["agents"][current_player]
        
        turn_info = {"turn": state.turn_count, "player": current_player}
        
        # Roll and move
        dice_roll = state.roll_dice()
        new_pos = state.move_player(current_player, dice_roll)
        turn_info["diceRoll"] = dice_roll
        turn_info["newPosition"] = new_pos
        
        # Handle landing
        if state.is_gamble_tile(new_pos):
            effect = random.choice(GAMBLE_EFFECTS)
            state.apply_gamble_effect(current_player, effect)
            turn_info["gambleEffect"] = effect.name
        else:
            prop = state.get_property_at(new_pos)
            if prop:
                if prop.owner is not None and prop.owner != current_player:
                    state.pay_fare(current_player, prop)
                elif prop.owner is None:
                    action = agent.choose_action(state)
                    if action == "BUY":
                        state.buy_property(current_player, prop)
                    turn_info["action"] = action
        
        state.current_player = 1 - current_player
        state.turn_count += 1
        state.check_game_over()
        
        turns_played.append(turn_info)
        game["history"].append(turn_info)
    
    return jsonify({
        "gameId": game_id,
        "state": state.to_dict(),
        "turnsPlayed": len(turns_played),
        "gameOver": state.game_over,
        "winner": state.winner
    })


@app.route('/api/game/<game_id>', methods=['DELETE'])
def delete_game(game_id: str):
    """Delete a game"""
    if game_id in active_games:
        del active_games[game_id]
        return jsonify({"message": "Game deleted"})
    return jsonify({"error": "Game not found"}), 404


@app.route('/api/simulate', methods=['POST'])
def simulate_tournament():
    """Run a simulation tournament between agents"""
    data = request.json or {}
    
    agent1_type = data.get("agent1", {}).get("type", "expectiminimax")
    agent1_config = data.get("agent1", {}).get("config", {})
    agent2_type = data.get("agent2", {}).get("type", "mcts")
    agent2_config = data.get("agent2", {}).get("config", {})
    num_games = min(data.get("numGames", 10), 100)  # Max 100 games
    
    try:
        agent1 = create_agent(agent1_type, 0, agent1_config)
        agent2 = create_agent(agent2_type, 1, agent2_config)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    
    results = {
        "agent1": {"type": agent1_type, "name": agent1.get_name(), "wins": 0},
        "agent2": {"type": agent2_type, "name": agent2.get_name(), "wins": 0},
        "totalGames": num_games,
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
        
        final_state, winner, _ = engine.play_game(verbose=False)
        actual_winner = player_map[winner] if winner is not None else None
        
        if actual_winner == 0:
            results["agent1"]["wins"] += 1
        elif actual_winner == 1:
            results["agent2"]["wins"] += 1
        
        results["games"].append({
            "gameNumber": game_num + 1,
            "winner": actual_winner,
            "turns": final_state.turn_count,
            "finalCash": final_state.cash
        })
    
    results["agent1"]["winRate"] = results["agent1"]["wins"] / num_games
    results["agent2"]["winRate"] = results["agent2"]["wins"] / num_games
    
    return jsonify(results)


if __name__ == '__main__':
    print("Starting AI Monopoly Python Backend...")
    print("Available endpoints:")
    print("  GET  /api/health - Health check")
    print("  GET  /api/agents - List available agents")
    print("  POST /api/game/new - Create new game")
    print("  GET  /api/game/<id>/state - Get game state")
    print("  POST /api/game/<id>/turn - Play one turn")
    print("  POST /api/game/<id>/play - Play full game")
    print("  POST /api/game/<id>/fast-forward - Play multiple turns")
    print("  POST /api/simulate - Run tournament simulation")
    
    app.run(host='0.0.0.0', port=5000, debug=True)
