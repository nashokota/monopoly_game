import React, { useState, useEffect, useCallback } from 'react';
import axios from 'axios';

const API_BASE = '/api';

// Agent options
const AGENT_OPTIONS = [
  { id: 'expectiminimax', name: 'Expectiminimax (Minimax)', description: 'Minimax with chance nodes for dice. Strategic and thorough.' },
  { id: 'minimax', name: 'Simplified Minimax', description: 'Faster minimax using average dice values.' },
  { id: 'mcts', name: 'Monte Carlo Tree Search', description: 'Simulation-based search. Great for uncertainty.' },
  { id: 'hybrid_mcts', name: 'Hybrid MCTS', description: 'MCTS with heuristic cutoff. Balanced approach.' }
];

// Color scheme for property groups (Dhaka city districts)
const COLOR_HEX_MAP = {
  'Brown': '#8B4513',
  'Light Blue': '#87CEEB',
  'Pink': '#FF69B4',
  'Orange': '#FF8C00',
  'Red': '#DC143C',
  'Yellow': '#FFD700',
  'Green': '#228B22',
  'Dark Blue': '#191970',
  'Purple': '#8B008B'
};

// Board position calculation for visual layout
const getBoardPosition = (index) => {
  // Bottom row: 0-10 (left to right)
  if (index <= 10) {
    return { row: 11, col: 11 - index };
  }
  // Left column: 11-19 (bottom to top)
  if (index <= 19) {
    return { row: 11 - (index - 10), col: 1 };
  }
  // Top row: 20-30 (left to right)
  if (index <= 30) {
    return { row: 1, col: index - 19 };
  }
  // Right column: 31-39 (top to bottom)
  return { row: index - 29, col: 11 };
};

// Tile component with distinct color styling
const Tile = ({ tile, players, currentPlayer }) => {
  const pos = getBoardPosition(tile.index);
  const player0Here = players[0] === tile.index;
  const player1Here = players[1] === tile.index;

  // Get the color for the property group
  const propertyColor = tile.color ? COLOR_HEX_MAP[tile.color] : '#444';
  
  // Determine background and border based on ownership
  let backgroundColor = 'rgba(255, 255, 255, 0.08)';
  let borderColor = propertyColor;
  let boxShadow = 'none';
  
  if (tile.owner === 0) {
    backgroundColor = 'rgba(0, 212, 255, 0.35)';
    borderColor = '#00d4ff';
    boxShadow = '0 0 12px rgba(0, 212, 255, 0.6), inset 0 0 20px rgba(0, 212, 255, 0.2)';
  } else if (tile.owner === 1) {
    backgroundColor = 'rgba(255, 107, 107, 0.35)';
    borderColor = '#ff6b6b';
    boxShadow = '0 0 12px rgba(255, 107, 107, 0.6), inset 0 0 20px rgba(255, 107, 107, 0.2)';
  } else if (tile.type === 'gamble') {
    backgroundColor = 'linear-gradient(135deg, #ffd700, #ff8c00)';
    borderColor = '#ffd700';
    boxShadow = '0 0 15px rgba(255, 215, 0, 0.5)';
  }

  const style = {
    gridRow: pos.row,
    gridColumn: pos.col,
    background: backgroundColor,
    borderLeft: tile.type === 'property' ? `5px solid ${propertyColor}` : 'none',
    borderTop: tile.owner !== null && tile.owner !== undefined ? `3px solid ${borderColor}` : 'none',
    boxShadow: boxShadow,
  };

  return (
    <div 
      className={`tile ${tile.type} ${tile.owner === 0 ? 'owned-blue' : ''} ${tile.owner === 1 ? 'owned-red' : ''}`}
      style={style}
      title={`${tile.name}${tile.color ? ` (${tile.color})` : ''}${tile.price ? ` - $${tile.price}` : ''}${tile.fare ? ` | Fare: $${tile.fare}` : ''}${tile.owner !== undefined && tile.owner !== null ? ` | Owned by Player ${tile.owner + 1}` : ''}`}
    >
      <div className="tile-color-bar" style={{ backgroundColor: propertyColor }}></div>
      <div className="tile-name">{tile.name}</div>
      {tile.price && <div className="tile-price">${tile.price}</div>}
      {tile.fare && <div className="tile-fare">🏠${tile.fare}</div>}
      {player0Here && <div className="player-marker player-0">🔵</div>}
      {player1Here && <div className="player-marker player-1">🔴</div>}
    </div>
  );
};

// Board component
const GameBoard = ({ board, positions, currentPlayer, turnCount, lastDice }) => {
  return (
    <div className="board-container">
      <div className="board">
        {board.map((tile, idx) => (
          <Tile 
            key={idx} 
            tile={tile} 
            players={positions}
            currentPlayer={currentPlayer}
          />
        ))}
        <div className="center-area">
          <div className="center-title">🏙️ DHAKA MONOPOLY</div>
          <div className="center-subtitle">AI Property Game</div>
          <div className="dice-display">🎲 {lastDice || '-'}</div>
          <div className="turn-info">
            Turn {turnCount}<br/>
            {currentPlayer === 0 ? '🔵' : '🔴'} Player {currentPlayer + 1}'s turn
          </div>
        </div>
      </div>
    </div>
  );
};

// Player Stats component
const PlayerStats = ({ playerStats, agents, currentPlayer }) => {
  return (
    <div className="panel-card">
      <h3>Player Stats</h3>
      <div className="player-stats">
        {playerStats.map((stats, idx) => (
          <div 
            key={idx} 
            className={`player-stat-row player-${idx} ${currentPlayer === idx ? 'active' : ''}`}
          >
            <div className={`stat-avatar player-${idx}`}>
              {idx === 0 ? '🔵' : '🔴'}
            </div>
            <div className="stat-info">
              <div className="stat-name">
                Player {idx + 1} {agents[idx]?.name ? `(${agents[idx].type})` : ''}
              </div>
              <div className="stat-details">
                {stats.propertyCount} properties • ${stats.propertyValue} value
              </div>
            </div>
            <div className="stat-cash">${stats.cash}</div>
          </div>
        ))}
      </div>
    </div>
  );
};

// Game Log component
const GameLog = ({ history }) => {
  return (
    <div className="panel-card">
      <h3>Game Log</h3>
      <div className="game-log">
        {history.slice(-10).reverse().map((entry, idx) => (
          <div key={idx} className={`log-entry player-${entry.player}`}>
            <span className="log-turn">T{entry.turn}</span> P{entry.player + 1}: 
            Rolled {entry.diceRoll} → Pos {entry.newPosition}
            {entry.action && ` [${entry.action}]`}
            {entry.farePaid && ` Paid $${entry.farePaid}`}
            {entry.gambleEffect && ` 🎰 ${entry.gambleEffect.name || entry.gambleEffect}`}
          </div>
        ))}
      </div>
    </div>
  );
};

// Game Over Modal
const GameOverModal = ({ winner, state, agents, onNewGame }) => {
  const winnerAgent = agents[winner];
  
  return (
    <div className="game-over-overlay">
      <div className="game-over-modal">
        <h2>🏆 Game Over!</h2>
        <div className="winner-display">
          <span className="player-name">
            Player {winner + 1} ({winnerAgent?.type || 'AI'})
          </span>
          <br />
          wins!
        </div>
        <div className="final-stats">
          <div className="final-stat-card">
            <h4>Player 1 Final</h4>
            <div className="value">${state.playerStats[0]?.cash || 0}</div>
          </div>
          <div className="final-stat-card">
            <h4>Player 2 Final</h4>
            <div className="value">${state.playerStats[1]?.cash || 0}</div>
          </div>
        </div>
        <button className="control-btn primary" onClick={onNewGame}>
          New Game
        </button>
      </div>
    </div>
  );
};

// Main App component
function App() {
  const [gameState, setGameState] = useState(null);
  const [gameId, setGameId] = useState(null);
  const [agents, setAgents] = useState([]);
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(false);
  const [autoPlay, setAutoPlay] = useState(false);
  const [gameOver, setGameOver] = useState(false);
  const [winner, setWinner] = useState(null);
  
  // Game setup state
  const [agent1Type, setAgent1Type] = useState('expectiminimax');
  const [agent2Type, setAgent2Type] = useState('mcts');
  
  // Simulation state
  const [simRunning, setSimRunning] = useState(false);
  const [simResults, setSimResults] = useState(null);
  const [simGames, setSimGames] = useState(10);

  // Start new game
  const startNewGame = async () => {
    setLoading(true);
    try {
      const response = await axios.post(`${API_BASE}/game/new`, {
        agent1: { type: agent1Type, config: {} },
        agent2: { type: agent2Type, config: {} },
        startingCash: 1500,
        maxTurns: 200
      });
      
      setGameId(response.data.gameId);
      setGameState(response.data.state);
      setAgents(response.data.agents);
      setHistory([]);
      setGameOver(false);
      setWinner(null);
    } catch (error) {
      console.error('Failed to start game:', error);
      alert('Failed to start game. Make sure the Python backend is running.');
    }
    setLoading(false);
  };

  // Play one turn
  const playTurn = useCallback(async () => {
    if (!gameId || gameOver) return;
    
    setLoading(true);
    try {
      const response = await axios.post(`${API_BASE}/game/${gameId}/turn`);
      setGameState(response.data.state);
      
      if (response.data.turnInfo) {
        setHistory(prev => [...prev, response.data.turnInfo]);
      }
      
      if (response.data.gameOver) {
        setGameOver(true);
        setWinner(response.data.winner);
        setAutoPlay(false);
      }
    } catch (error) {
      console.error('Failed to play turn:', error);
      setAutoPlay(false);
    }
    setLoading(false);
  }, [gameId, gameOver]);

  // Auto-play effect
  useEffect(() => {
    if (autoPlay && !loading && !gameOver) {
      const timer = setTimeout(playTurn, 500);
      return () => clearTimeout(timer);
    }
  }, [autoPlay, loading, gameOver, playTurn]);

  // Fast forward
  const fastForward = async (turns = 10) => {
    if (!gameId || gameOver) return;
    
    setLoading(true);
    try {
      const response = await axios.post(`${API_BASE}/game/${gameId}/fast-forward`, {
        turns
      });
      setGameState(response.data.state);
      
      if (response.data.gameOver) {
        setGameOver(true);
        setWinner(response.data.winner);
      }
    } catch (error) {
      console.error('Failed to fast forward:', error);
    }
    setLoading(false);
  };

  // Play to completion
  const playToEnd = async () => {
    if (!gameId || gameOver) return;
    
    setLoading(true);
    try {
      const response = await axios.post(`${API_BASE}/game/${gameId}/play`);
      setGameState(response.data.state);
      setHistory(response.data.history || []);
      setGameOver(true);
      setWinner(response.data.winner);
    } catch (error) {
      console.error('Failed to complete game:', error);
    }
    setLoading(false);
  };

  // Run simulation
  const runSimulation = async () => {
    setSimRunning(true);
    try {
      const response = await axios.post(`${API_BASE}/simulate`, {
        agent1: { type: agent1Type, config: {} },
        agent2: { type: agent2Type, config: {} },
        numGames: simGames
      });
      setSimResults(response.data);
    } catch (error) {
      console.error('Failed to run simulation:', error);
      alert('Failed to run simulation. Make sure the Python backend is running.');
    }
    setSimRunning(false);
  };

  // Get board from state
  const board = gameState?.board || [];
  const positions = gameState?.positions || [0, 0];
  const playerStats = gameState?.playerStats || [
    { cash: 1500, propertyCount: 0, propertyValue: 0 },
    { cash: 1500, propertyCount: 0, propertyValue: 0 }
  ];

  return (
    <div className="app">
      <header className="header">
        <h1>🏙️ DHAKA MONOPOLY</h1>
        <p>Watch two AI agents compete to own Dhaka's famous neighborhoods!</p>
      </header>

      {!gameId ? (
        <>
          {/* Game Setup */}
          <div className="game-setup">
            <div className="agent-selection">
              <div className="agent-card player-0">
                <h3><span className="player-dot player-0"></span> Player 1 (Blue)</h3>
                <select 
                  value={agent1Type} 
                  onChange={(e) => setAgent1Type(e.target.value)}
                >
                  {AGENT_OPTIONS.map(agent => (
                    <option key={agent.id} value={agent.id}>{agent.name}</option>
                  ))}
                </select>
                <p className="agent-description">
                  {AGENT_OPTIONS.find(a => a.id === agent1Type)?.description}
                </p>
              </div>
              
              <div className="agent-card player-1">
                <h3><span className="player-dot player-1"></span> Player 2 (Red)</h3>
                <select 
                  value={agent2Type} 
                  onChange={(e) => setAgent2Type(e.target.value)}
                >
                  {AGENT_OPTIONS.map(agent => (
                    <option key={agent.id} value={agent.id}>{agent.name}</option>
                  ))}
                </select>
                <p className="agent-description">
                  {AGENT_OPTIONS.find(a => a.id === agent2Type)?.description}
                </p>
              </div>
            </div>
            
            <button 
              className="start-button" 
              onClick={startNewGame}
              disabled={loading}
            >
              {loading ? 'Starting...' : 'Start Game'}
            </button>
          </div>

          {/* Simulation Panel */}
          <div className="simulation-panel">
            <h3>🧪 Tournament Simulation</h3>
            <div className="sim-controls">
              <span>Run</span>
              <input 
                type="number" 
                value={simGames} 
                onChange={(e) => setSimGames(Math.min(100, Math.max(1, parseInt(e.target.value) || 1)))}
                min="1" 
                max="100"
              />
              <span>games</span>
              <button 
                className="control-btn primary"
                onClick={runSimulation}
                disabled={simRunning}
              >
                {simRunning ? 'Running...' : 'Run Simulation'}
              </button>
            </div>
            
            {simResults && (
              <div className="sim-results">
                <div className="sim-result-card">
                  <h4>{simResults.agent1.name}</h4>
                  <div className={`win-rate ${simResults.agent1.winRate > 0.5 ? 'high' : 'low'}`}>
                    {(simResults.agent1.winRate * 100).toFixed(1)}%
                  </div>
                  <div>{simResults.agent1.wins} wins</div>
                </div>
                <div className="sim-result-card">
                  <h4>{simResults.agent2.name}</h4>
                  <div className={`win-rate ${simResults.agent2.winRate > 0.5 ? 'high' : 'low'}`}>
                    {(simResults.agent2.winRate * 100).toFixed(1)}%
                  </div>
                  <div>{simResults.agent2.wins} wins</div>
                </div>
              </div>
            )}
          </div>
        </>
      ) : (
        /* Game View */
        <div className="game-container">
          <GameBoard 
            board={board}
            positions={positions}
            currentPlayer={gameState?.currentPlayer || 0}
            turnCount={gameState?.turnCount || 0}
            lastDice={gameState?.lastDiceRoll}
          />
          
          <div className="side-panel">
            <PlayerStats 
              playerStats={playerStats}
              agents={agents}
              currentPlayer={gameState?.currentPlayer || 0}
            />
            
            <div className="panel-card">
              <h3>Controls</h3>
              <div className="game-controls">
                <button 
                  className="control-btn primary"
                  onClick={playTurn}
                  disabled={loading || gameOver || autoPlay}
                >
                  Play Turn
                </button>
                <button 
                  className="control-btn secondary"
                  onClick={() => setAutoPlay(!autoPlay)}
                  disabled={loading || gameOver}
                >
                  {autoPlay ? '⏸ Pause' : '▶ Auto-Play'}
                </button>
                <button 
                  className="control-btn secondary"
                  onClick={() => fastForward(10)}
                  disabled={loading || gameOver || autoPlay}
                >
                  ⏩ Skip 10 Turns
                </button>
                <button 
                  className="control-btn secondary"
                  onClick={playToEnd}
                  disabled={loading || gameOver || autoPlay}
                >
                  ⏭ Play to End
                </button>
                <button 
                  className="control-btn danger"
                  onClick={() => {
                    setGameId(null);
                    setGameState(null);
                    setGameOver(false);
                    setAutoPlay(false);
                  }}
                >
                  New Game
                </button>
              </div>
            </div>
            
            <GameLog history={history} />
          </div>
        </div>
      )}

      {/* Game Over Modal */}
      {gameOver && winner !== null && (
        <GameOverModal 
          winner={winner}
          state={gameState}
          agents={agents}
          onNewGame={() => {
            setGameId(null);
            setGameState(null);
            setGameOver(false);
            setWinner(null);
          }}
        />
      )}

      {/* Loading overlay for long operations */}
      {loading && simRunning && (
        <div className="game-over-overlay">
          <div className="loading">
            <div className="spinner"></div>
            <p style={{ marginTop: '20px' }}>Running simulation...</p>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
