import express from 'express';
import multer from 'multer';
import sqlite3 from 'sqlite3';
import { open } from 'sqlite';
import { Server } from 'socket.io';
import { createServer } from 'http';
import path from 'path';
import { fileURLToPath } from 'url';
import cors from 'cors';
import fs from 'fs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const httpServer = createServer(app);
const io = new Server(httpServer, {
  cors: { origin: "*" }
});

app.use(cors());
app.use(express.json());

// Configure multer for file uploads
const upload = multer({ storage: multer.memoryStorage() });

// Initialize SQLite database
let db;
async function initDatabase() {
  const dataDir = path.join(__dirname, 'data');
  
  // Ensure data directory exists
  if (!fs.existsSync(dataDir)) {
    fs.mkdirSync(dataDir, { recursive: true });
    console.log('📁 Created data directory:', dataDir);
  }
  
  const dbPath = path.join(dataDir, 'camera.db');
  console.log('📂 Database path:', dbPath);
  
  try {
    db = await open({
      filename: dbPath,
      driver: sqlite3.Database
    });
    
    // Create tables
    await db.exec(`
      CREATE TABLE IF NOT EXISTS detections (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp INTEGER NOT NULL,
        probability REAL NOT NULL,
        result TEXT NOT NULL,
        image_data BLOB
      );
      
      CREATE INDEX IF NOT EXISTS idx_timestamp ON detections(timestamp);
      CREATE INDEX IF NOT EXISTS idx_result ON detections(result);
    `);
    
    console.log('✅ Database initialized successfully');
    
    // Test database
    const test = await db.get('SELECT COUNT(*) as count FROM detections');
    console.log(`📊 Existing records: ${test.count}`);
    
  } catch (error) {
    console.error('❌ Database error:', error);
    throw error;
  }
}

// HTTP endpoint that receives everything from ESP32
app.post('/upload-inference', upload.fields([
  { name: 'metadata' },
  { name: 'image' }
]), async (req, res) => {
  try {
    // Parse metadata from ESP32
    const metadata = JSON.parse(req.body.metadata);
    const imageBuffer = req.files.image[0].buffer;
    
    console.log(`\n📸 Received from ESP32:`);
    console.log(`   Result: ${metadata.result}`);
    console.log(`   Probability: ${(metadata.probability * 100).toFixed(1)}%`);
    console.log(`   Image size: ${imageBuffer.length} bytes`);
    console.log(`   Timestamp: ${new Date(metadata.timestamp).toLocaleTimeString()}`);
    
    // Save to database
    const result = await db.run(
      'INSERT INTO detections (timestamp, probability, result, image_data) VALUES (?, ?, ?, ?)',
      metadata.timestamp, metadata.probability, metadata.result, imageBuffer
    );
    
    console.log(`   Saved to database with ID: ${result.lastID}`);
    
    // Broadcast to web clients via WebSocket
    io.emit('new-detection', {
      id: result.lastID,
      timestamp: metadata.timestamp,
      probability: metadata.probability,
      result: metadata.result
    });
    
    res.json({ success: true, id: result.lastID });
    
  } catch (error) {
    console.error('❌ Error processing upload:', error);
    res.status(500).json({ error: error.message });
  }
});

// API endpoint for web frontend to get detections
app.get('/api/detections', async (req, res) => {
  try {
    const limit = parseInt(req.query.limit) || 50;
    const detections = await db.all(
      'SELECT id, timestamp, probability, result FROM detections ORDER BY timestamp DESC LIMIT ?',
      limit
    );
    res.json(detections);
  } catch (error) {
    console.error('Error fetching detections:', error);
    res.status(500).json({ error: error.message });
  }
});

// API endpoint to get image for a detection
app.get('/api/image/:id', async (req, res) => {
  try {
    const result = await db.get(
      'SELECT image_data FROM detections WHERE id = ?',
      req.params.id
    );
    if (result && result.image_data) {
      res.set('Content-Type', 'image/jpeg');
      res.send(result.image_data);
    } else {
      res.status(404).json({ error: 'Image not found' });
    }
  } catch (error) {
    console.error('Error fetching image:', error);
    res.status(500).json({ error: error.message });
  }
});


// API endpoint for statistics
app.get('/api/statistics', async (req, res) => {
  try {
    const hours = parseInt(req.query.hours) || 24;
    const since = Date.now() - (hours * 3600000);
    
    // Get all detections in time range
    const allDetections = await db.all(`
      SELECT result, probability, timestamp
      FROM detections
      WHERE timestamp > ?
      ORDER BY timestamp DESC
    `, since);
    
    const total_detections = allDetections.length;
    const not_allowed_count = allDetections.filter(d => d.result === 'NOT_ALLOWED').length;
    const avg_probability = allDetections.reduce((sum, d) => sum + d.probability, 0) / total_detections || 0;
    
    // Get hourly breakdown
    const hourly = await db.all(`
      SELECT 
        strftime('%H:00', datetime(timestamp/1000, 'unixepoch', 'localtime')) as hour,
        COUNT(*) as count,
        SUM(CASE WHEN result = 'NOT_ALLOWED' THEN 1 ELSE 0 END) as violations
      FROM detections
      WHERE timestamp > ?
      GROUP BY hour
      ORDER BY hour
    `, since);
    
    const stats = {
      total_detections,
      not_allowed_count,
      avg_probability,
      min_probability: Math.min(...allDetections.map(d => d.probability), 0),
      max_probability: Math.max(...allDetections.map(d => d.probability), 0)
    };
    
    res.json({ stats, hourly });
  } catch (error) {
    console.error('Error fetching statistics:', error);
    res.status(500).json({ error: error.message });
  }
});

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({ 
    status: 'ok', 
    timestamp: Date.now(),
    database: db ? 'connected' : 'disconnected'
  });
});

// WebSocket connection
io.on('connection', (socket) => {
  console.log('🔌 Web client connected');
  socket.on('disconnect', () => {
    console.log('🔌 Web client disconnected');
  });
});

// Start server
const PORT = 8080;
initDatabase().then(() => {
  httpServer.listen(PORT, '0.0.0.0', () => {
    console.log(`\n✅ Server running on port ${PORT}`);
    console.log(`   POST endpoint: http://localhost:${PORT}/upload-inference`);
    console.log(`   API: http://localhost:${PORT}/api/detections`);
    console.log(`   Health: http://localhost:${PORT}/health`);
    console.log(`\n📡 Waiting for ESP32 connection...\n`);
  });
}).catch(error => {
  console.error('Failed to initialize database:', error);
  process.exit(1);
});
