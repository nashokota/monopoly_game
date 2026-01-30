# AI Monopoly

A simplified Monopoly-style game where two AI agents compete using different algorithms.

## Game Rules

- **40 boxes** total: 36 properties (9 colors × 4 each) + 4 gambling tiles
- Each agent starts with **$1500**
- Roll dice (2-12) to move around the board
- **Buy** unowned properties or **skip**
- **Pay fare** when landing on opponent's property (double if they own all of that color)
- **Gambling tiles** trigger random effects (win/lose money, pay/receive from opponent)
- Game ends when all properties are sold and turn limit reached, or a player goes bankrupt
- **Winner** is the player with more total wealth (cash + property value)

## AI Agents

### 1. Expectiminimax (Minimax with Chance Nodes)
- Handles dice randomness using expectation over outcomes
- Uses alpha-beta pruning for efficiency
- Strategic lookahead with depth-limited search
- Best for thorough strategic analysis

### 2. Monte Carlo Tree Search (MCTS)
- Simulation-based approach
- Uses UCB1 for exploration/exploitation balance
- Naturally handles uncertainty through random playouts
- Great for handling complex random elements

## Project Structure

```
ai-monopoly/
├── client/                 # React frontend
│   ├── public/
│   ├── src/
│   │   ├── App.js         # Main game UI
│   │   ├── index.js
│   │   └── index.css      # Styling
│   └── package.json
│
├── server/                 # Node.js/Express backend
│   ├── index.js           # API routes
│   └── package.json
│
└── python-ai/             # Python AI agents
    ├── app.py             # Flask API server
    ├── game_state.py      # Game logic
    ├── minimax_agent.py   # Expectiminimax implementation
    ├── mcts_agent.py      # MCTS implementation
    ├── game_engine.py     # Game runner
    └── requirements.txt
```

## Setup & Installation

### Prerequisites
- Node.js 18+
- Python 3.9+
- MongoDB (optional, for game history)

### 1. Install Python dependencies

```bash
cd python-ai
pip install -r requirements.txt
```

### 2. Install Node.js dependencies

```bash
# Server
cd server
npm install

# Client
cd ../client
npm install
```

### 3. Start the services

**Terminal 1 - Python AI Backend:**
```bash
cd python-ai
python app.py
```
Runs on http://localhost:5000

**Terminal 2 - Node.js Backend:**
```bash
cd server
npm start
```
Runs on http://localhost:3001

**Terminal 3 - React Frontend:**
```bash
cd client
npm start
```
Runs on http://localhost:3000

## API Endpoints

### Node.js Backend (proxies to Python)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Health check |
| GET | `/api/agents` | List available AI agents |
| GET | `/api/board` | Get board configuration |
| POST | `/api/game/new` | Create new game |
| GET | `/api/game/:id/state` | Get current game state |
| POST | `/api/game/:id/turn` | Play single turn |
| POST | `/api/game/:id/fast-forward` | Play multiple turns |
| POST | `/api/game/:id/play` | Play game to completion |
| POST | `/api/simulate` | Run tournament simulation |

## How the Algorithms Work

### Expectiminimax

```
function expectiminimax(state, depth, isMax):
    if terminal or depth == 0:
        return evaluate(state)
    
    if chance_node:  # Dice roll or gambling
        return average over all outcomes
    
    if isMax:
        return max over all actions
    else:
        return min over all actions
```

The evaluation function considers:
- Cash difference between players
- Property value difference
- Potential fare income
- Monopoly bonuses (owning all of a color)

### Monte Carlo Tree Search

```
for simulation in range(num_simulations):
    1. SELECT: Use UCB1 to find promising node
    2. EXPAND: Add new child for untried action
    3. SIMULATE: Random playout to terminal/cutoff
    4. BACKPROPAGATE: Update statistics up the tree
```

UCB1 formula: `exploitation + C * sqrt(ln(parent.visits) / child.visits)`

## Configuration

### Agent Settings

**Expectiminimax:**
- `depth`: Search depth (default: 4)
- `use_sampling`: Sample dice outcomes (default: true)
- `samples`: Number of dice samples (default: 5)

**MCTS:**
- `simulations`: Number of simulations (default: 500)
- `exploration`: UCB1 exploration constant (default: 1.414)
- `max_depth`: Max simulation depth (default: 50)

## Screenshots

The game features:
- Visual board with property colors
- Player position markers
- Real-time stats display
- Turn-by-turn game log
- Tournament simulation mode

## License

MIT
