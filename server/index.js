const express = require('express');
const cors = require('cors');
const mongoose = require('mongoose');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3001;
const PYTHON_API = process.env.PYTHON_API || 'http://localhost:5002';

// Middleware
app.use(cors());
app.use(express.json());

// MongoDB connection (optional - for storing game history)
const MONGODB_URI = process.env.MONGODB_URI || 'mongodb://localhost:27017/ai-monopoly';

mongoose.connect(MONGODB_URI)
  .then(() => console.log('Connected to MongoDB'))
  .catch(err => console.log('MongoDB connection optional:', err.message));

// Game History Schema
const gameHistorySchema = new mongoose.Schema({
  gameId: { type: String, required: true, unique: true },
  agent1: {
    type: { type: String },
    name: String,
    config: Object
  },
  agent2: {
    type: { type: String },
    name: String,
    config: Object
  },
  winner: Number,
  totalTurns: Number,
  finalState: Object,
  history: Array,
  createdAt: { type: Date, default: Date.now },
  completedAt: Date
});

const GameHistory = mongoose.model('GameHistory', gameHistorySchema);

// Tournament Results Schema
const tournamentSchema = new mongoose.Schema({
  agent1: Object,
  agent2: Object,
  totalGames: Number,
  results: Object,
  createdAt: { type: Date, default: Date.now }
});

const Tournament = mongoose.model('Tournament', tournamentSchema);

// Health check
app.get('/api/health', async (req, res) => {
  try {
    const pythonHealth = await axios.get(`${PYTHON_API}/api/health`);
    res.json({
      status: 'healthy',
      service: 'AI Monopoly Node.js Backend',
      pythonBackend: pythonHealth.data
    });
  } catch (error) {
    res.json({
      status: 'healthy',
      service: 'AI Monopoly Node.js Backend',
      pythonBackend: { status: 'unavailable' }
    });
  }
});

// Get available agents (proxy to Python)
app.get('/api/agents', async (req, res) => {
  try {
    const response = await axios.get(`${PYTHON_API}/api/agents`);
    res.json(response.data);
  } catch (error) {
    res.status(500).json({ error: 'Failed to fetch agents from Python backend' });
  }
});

// Create new game
app.post('/api/game/new', async (req, res) => {
  try {
    const response = await axios.post(`${PYTHON_API}/api/game/new`, req.body);
    const gameData = response.data;
    
    // Save to MongoDB if connected
    try {
      await GameHistory.create({
        gameId: gameData.gameId,
        agent1: gameData.agents[0],
        agent2: gameData.agents[1],
        finalState: gameData.state,
        history: []
      });
    } catch (dbError) {
      console.log('Could not save to MongoDB:', dbError.message);
    }
    
    res.json(gameData);
  } catch (error) {
    res.status(500).json({ error: error.response?.data?.error || 'Failed to create game' });
  }
});

// Get game state
app.get('/api/game/:gameId/state', async (req, res) => {
  try {
    const response = await axios.get(`${PYTHON_API}/api/game/${req.params.gameId}/state`);
    res.json(response.data);
  } catch (error) {
    if (error.response?.status === 404) {
      return res.status(404).json({ error: 'Game not found' });
    }
    res.status(500).json({ error: 'Failed to fetch game state' });
  }
});

// Play single turn
app.post('/api/game/:gameId/turn', async (req, res) => {
  try {
    const response = await axios.post(`${PYTHON_API}/api/game/${req.params.gameId}/turn`);
    const turnData = response.data;
    
    // Update MongoDB if game is over
    if (turnData.gameOver) {
      try {
        await GameHistory.findOneAndUpdate(
          { gameId: req.params.gameId },
          {
            winner: turnData.winner,
            totalTurns: turnData.state.turnCount,
            finalState: turnData.state,
            completedAt: new Date()
          }
        );
      } catch (dbError) {
        console.log('Could not update MongoDB:', dbError.message);
      }
    }
    
    res.json(turnData);
  } catch (error) {
    if (error.response?.status === 404) {
      return res.status(404).json({ error: 'Game not found' });
    }
    res.status(500).json({ error: 'Failed to play turn' });
  }
});

// Play full game
app.post('/api/game/:gameId/play', async (req, res) => {
  try {
    const response = await axios.post(`${PYTHON_API}/api/game/${req.params.gameId}/play`);
    const gameData = response.data;
    
    // Save to MongoDB
    try {
      await GameHistory.findOneAndUpdate(
        { gameId: req.params.gameId },
        {
          winner: gameData.winner,
          totalTurns: gameData.totalTurns,
          finalState: gameData.state,
          history: gameData.history,
          completedAt: new Date()
        }
      );
    } catch (dbError) {
      console.log('Could not update MongoDB:', dbError.message);
    }
    
    res.json(gameData);
  } catch (error) {
    res.status(500).json({ error: 'Failed to play full game' });
  }
});

// Fast forward multiple turns
app.post('/api/game/:gameId/fast-forward', async (req, res) => {
  try {
    const response = await axios.post(
      `${PYTHON_API}/api/game/${req.params.gameId}/fast-forward`,
      req.body
    );
    res.json(response.data);
  } catch (error) {
    res.status(500).json({ error: 'Failed to fast forward' });
  }
});

// Delete game
app.delete('/api/game/:gameId', async (req, res) => {
  try {
    await axios.delete(`${PYTHON_API}/api/game/${req.params.gameId}`);
    
    // Also delete from MongoDB
    try {
      await GameHistory.deleteOne({ gameId: req.params.gameId });
    } catch (dbError) {
      console.log('Could not delete from MongoDB:', dbError.message);
    }
    
    res.json({ message: 'Game deleted' });
  } catch (error) {
    res.status(500).json({ error: 'Failed to delete game' });
  }
});

// Run tournament simulation
app.post('/api/simulate', async (req, res) => {
  try {
    const response = await axios.post(`${PYTHON_API}/api/simulate`, req.body);
    const results = response.data;
    
    // Save tournament results
    try {
      await Tournament.create({
        agent1: results.agent1,
        agent2: results.agent2,
        totalGames: results.totalGames,
        results: results
      });
    } catch (dbError) {
      console.log('Could not save tournament to MongoDB:', dbError.message);
    }
    
    res.json(results);
  } catch (error) {
    res.status(500).json({ error: 'Failed to run simulation' });
  }
});

// Get game history from MongoDB
app.get('/api/history', async (req, res) => {
  try {
    const games = await GameHistory.find()
      .sort({ createdAt: -1 })
      .limit(50)
      .select('-history');
    res.json({ games });
  } catch (error) {
    res.json({ games: [], error: 'Database not available' });
  }
});

// Get specific game from history
app.get('/api/history/:gameId', async (req, res) => {
  try {
    const game = await GameHistory.findOne({ gameId: req.params.gameId });
    if (!game) {
      return res.status(404).json({ error: 'Game not found in history' });
    }
    res.json(game);
  } catch (error) {
    res.status(500).json({ error: 'Failed to fetch game history' });
  }
});

// Get tournament history
app.get('/api/tournaments', async (req, res) => {
  try {
    const tournaments = await Tournament.find()
      .sort({ createdAt: -1 })
      .limit(20);
    res.json({ tournaments });
  } catch (error) {
    res.json({ tournaments: [], error: 'Database not available' });
  }
});

// Board configuration endpoint
app.get('/api/board', (req, res) => {
  // Return the board configuration for the frontend
  const colors = [
    { name: 'Brown', basePrice: 60, baseFare: 30, hex: '#8B4513' },
    { name: 'Light Blue', basePrice: 80, baseFare: 40, hex: '#87CEEB' },
    { name: 'Pink', basePrice: 100, baseFare: 50, hex: '#FFB6C1' },
    { name: 'Orange', basePrice: 120, baseFare: 60, hex: '#FFA500' },
    { name: 'Red', basePrice: 150, baseFare: 75, hex: '#FF0000' },
    { name: 'Yellow', basePrice: 180, baseFare: 90, hex: '#FFD700' },
    { name: 'Green', basePrice: 220, baseFare: 110, hex: '#228B22' },
    { name: 'Dark Blue', basePrice: 280, baseFare: 140, hex: '#00008B' },
    { name: 'Purple', basePrice: 320, baseFare: 160, hex: '#800080' },
  ];
  
  const gamblePositions = [9, 19, 29, 39];
  
  const board = [];
  let propertyIdx = 0;
  
  for (let i = 0; i < 40; i++) {
    if (gamblePositions.includes(i)) {
      board.push({
        index: i,
        type: 'gamble',
        name: `Gamble Zone ${gamblePositions.indexOf(i) + 1}`,
        hex: '#FFD700'
      });
    } else {
      const colorIdx = Math.floor(propertyIdx / 4);
      const posInColor = propertyIdx % 4;
      const color = colors[colorIdx];
      
      board.push({
        index: i,
        type: 'property',
        name: `${color.name} Property ${posInColor + 1}`,
        color: color.name,
        hex: color.hex,
        price: color.basePrice + (posInColor * 10),
        fare: color.baseFare + (posInColor * 5)
      });
      
      propertyIdx++;
    }
  }
  
  res.json({ board, colors, gamblePositions });
});

app.listen(PORT, () => {
  console.log(`AI Monopoly Node.js Backend running on port ${PORT}`);
  console.log(`Python API expected at: ${PYTHON_API}`);
});
