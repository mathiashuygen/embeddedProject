import sqlite3 from 'sqlite3';
import { open } from 'sqlite';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

let db;

export async function initDatabase() {
  db = await open({
    filename: join(__dirname, '../data/camera.db'),
    driver: sqlite3.Database
  });

  await db.exec(`
    CREATE TABLE IF NOT EXISTS detections (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      timestamp INTEGER NOT NULL,
      probability REAL NOT NULL,
      result TEXT NOT NULL,
      has_image INTEGER DEFAULT 0,
      image_path TEXT
    );

    CREATE TABLE IF NOT EXISTS images (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      detection_id INTEGER,
      image_data BLOB,
      created_at INTEGER,
      FOREIGN KEY(detection_id) REFERENCES detections(id)
    );

    CREATE INDEX IF NOT EXISTS idx_timestamp ON detections(timestamp);
    CREATE INDEX IF NOT EXISTS idx_result ON detections(result);
  `);

  console.log('Database initialized');
  return db;
}

export async function saveDetection(probability, result, imageBuffer = null) {
  const timestamp = Date.now();
  const result_stmt = await db.prepare(
    'INSERT INTO detections (timestamp, probability, result, has_image) VALUES (?, ?, ?, ?)'
  );
  const info = await result_stmt.run(timestamp, probability, result, imageBuffer ? 1 : 0);
  await result_stmt.finalize();

  if (imageBuffer) {
    const img_stmt = await db.prepare(
      'INSERT INTO images (detection_id, image_data, created_at) VALUES (?, ?, ?)'
    );
    await img_stmt.run(info.lastID, imageBuffer, timestamp);
    await img_stmt.finalize();
  }

  return info.lastID;
}

export async function getStatistics(hours = 24) {
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
      strftime('%H:00', datetime(timestamp/1000, 'unixepoch')) as hour,
      COUNT(*) as count,
      SUM(CASE WHEN result = 'NOT_ALLOWED' THEN 1 ELSE 0 END) as violations
    FROM detections
    WHERE timestamp > ?
    GROUP BY hour
    ORDER BY hour
  `, since);

  return { stats, hourly };
}

export async function getRecentDetections(limit = 100) {
  return await db.all(`
    SELECT d.*, i.image_data
    FROM detections d
    LEFT JOIN images i ON d.id = i.detection_id
    ORDER BY d.timestamp DESC
    LIMIT ?
  `, limit);
}

export async function getImage(detectionId) {
  const image = await db.get(
    'SELECT image_data FROM images WHERE detection_id = ?',
    detectionId
  );
  return image ? image.image_data : null;
}

export default { initDatabase, saveDetection, getStatistics, getRecentDetections, getImage };
