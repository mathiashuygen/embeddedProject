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
import sharp from 'sharp';

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

// Helper: Convert RGB565 to RGB888 buffer
function rgb565ToRgb888(rgb565Buffer) {
  const pixelCount = rgb565Buffer.length / 2;
  const rgb888 = Buffer.alloc(pixelCount * 3);
  
  for (let i = 0; i < pixelCount; i++) {
    // Read RGB565 value (little-endian: low byte then high byte)
    const highByte = rgb565Buffer[i * 2];
    const lowByte = rgb565Buffer[i * 2 + 1];
    const rgb565 = (highByte << 8) | lowByte;    
    // Extract and expand components
    let r = (rgb565 >> 11) & 0x1F;
    let g = (rgb565 >> 5) & 0x3F;
    let b = rgb565 & 0x1F;
    
    // Expand to 8-bit
    r = (r << 3) | (r >> 2);
    g = (g << 2) | (g >> 4);
    b = (b << 3) | (b >> 2);
    
    rgb888[i * 3] = r;
    rgb888[i * 3 + 1] = g;
    rgb888[i * 3 + 2] = b;
  }
  
  return rgb888;
}

// Helper: Convert RGB565 buffer to JPEG
async function rgb565ToJpeg(rgb565Buffer, width = 96, height = 96) {
  const rgb888 = rgb565ToRgb888(rgb565Buffer);
  const imageBuffer = await sharp(rgb888, {
    raw: {
      width: width,
      height: height,
      channels: 3
    }
  }).jpeg({ quality: 85 }).toBuffer();
  return imageBuffer;
}

// Initialize SQLite database
let db;
async function initDatabase() {
  const dataDir = path.join(__dirname, 'data');
  
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
    const rawImageBuffer = req.files.image[0].buffer;
    
    console.log(`\n📸 Received from ESP32:`);
    console.log(`   Result: ${metadata.result}`);
    console.log(`   Probability: ${(metadata.probability * 100).toFixed(1)}%`);
    console.log(`   Raw image size: ${rawImageBuffer.length} bytes`);
    
    // Check if it's RGB565 format (96x96x2 = 18432 bytes)
    let jpegBuffer;
    if (rawImageBuffer.length === 96 * 96 * 2) {
      console.log(`   Detected RGB565 format, converting to JPEG...`);
      jpegBuffer = await rgb565ToJpeg(rawImageBuffer, 96, 96);
      console.log(`   Converted to JPEG: ${jpegBuffer.length} bytes`);
    } else if (rawImageBuffer[0] === 0xFF && rawImageBuffer[1] === 0xD8) {
      console.log(`   Already JPEG format`);
      jpegBuffer = rawImageBuffer;
    } else {
      console.log(`   Unknown format, saving as is`);
      jpegBuffer = rawImageBuffer;
    }
    
    // Save to database
    const result = await db.run(
      'INSERT INTO detections (timestamp, probability, result, image_data) VALUES (?, ?, ?, ?)',
      metadata.timestamp, metadata.probability, metadata.result, jpegBuffer
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
    res.set('Cache-Control', 'no-store');  // no caching
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
    
    const stats = await db.get(`
      SELECT 
        COUNT(*) as total_detections,
        SUM(CASE WHEN result = 'NOT_ALLOWED' THEN 1 ELSE 0 END) as not_allowed_count,
        AVG(probability) as avg_probability,
        MIN(probability) as min_probability,
        MAX(probability) as max_probability
      FROM detections
      WHERE timestamp > ?
    `, since);
    
    const hourly = await db.all(`
      SELECT 
        strftime('%H', datetime(timestamp/1000, 'unixepoch')) as utc_hour,
        MIN(timestamp) as sample_ts,
        COUNT(*) as count,
        SUM(CASE WHEN result = 'NOT_ALLOWED' THEN 1 ELSE 0 END) as violations
      FROM detections
      WHERE timestamp > ?
      GROUP BY utc_hour
      ORDER BY utc_hour
    `, since);

    const localHourly = hourly.map(row => {
      const date = new Date(row.sample_ts);
      const localHour = date.toLocaleTimeString('en-GB', {
        hour: '2-digit', minute: '2-digit', timeZone: 'Europe/Brussels'
      }).slice(0, 5);
      return {
        hour: localHour,
        count: row.count,
        violations: row.violations,
      };
    });

    res.set('Cache-Control', 'no-store');
    res.json({ stats, hourly: localHourly });

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
  console.log('Web client connected');
  socket.on('disconnect', () => {
    console.log('Web client disconnected');
  });
});


let sleepModeEnabled = true;

app.get('/api/sleep-mode', (req, res) => {
  res.json({ enabled: sleepModeEnabled });
});

app.post('/api/sleep-mode', (req, res) => {
  sleepModeEnabled = req.body.enabled;
  res.json({ enabled: sleepModeEnabled });
});

// Start server
const PORT = 8080;
initDatabase().then(() => {
  httpServer.listen(PORT, '0.0.0.0', () => {
    console.log(`\nServer running on port ${PORT}`);
    console.log(`   POST endpoint: http://localhost:${PORT}/upload-inference`);
    console.log(`   API: http://localhost:${PORT}/api/detections`);
    console.log(`   Health: http://localhost:${PORT}/health`);
    console.log(`\nWaiting for ESP32 connection...\n`);
  });
}).catch(error => {
  console.error('Failed to initialize database:', error);
  process.exit(1);
});
