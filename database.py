import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Optional
import os

class PlaceDatabase:
    """SQLite database for managing place metadata and statistics"""
    
    def __init__(self, db_path="room_memory/places.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.init_db()
    
    def init_db(self):
        """Initialize database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Places table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS places (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                description TEXT,
                image_count INTEGER DEFAULT 1,
                recognition_count INTEGER DEFAULT 0,
                last_recognized TIMESTAMP,
                avg_confidence REAL DEFAULT 0.0
            )
        ''')
        
        # Recognition history table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS recognition_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                place_name TEXT NOT NULL,
                confidence REAL NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (place_name) REFERENCES places(name) ON DELETE CASCADE
            )
        ''')
        
        # Place images table (for multiple images per place)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS place_images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                place_name TEXT NOT NULL,
                image_path TEXT NOT NULL,
                added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (place_name) REFERENCES places(name) ON DELETE CASCADE
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def add_place(self, name: str, image_path: str, description: str = ""):
        """Add a new place to the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Insert or update place
            cursor.execute('''
                INSERT INTO places (name, description, image_count)
                VALUES (?, ?, 1)
                ON CONFLICT(name) DO UPDATE SET
                    image_count = image_count + 1,
                    updated_at = CURRENT_TIMESTAMP
            ''', (name, description))
            
            # Add image reference
            cursor.execute('''
                INSERT INTO place_images (place_name, image_path)
                VALUES (?, ?)
            ''', (name, image_path))
            
            conn.commit()
            return True
        except Exception as e:
            print(f"Error adding place: {e}")
            conn.rollback()
            return False
        finally:
            conn.close()
    
    def delete_place(self, name: str):
        """Delete a place and all its data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('DELETE FROM places WHERE name = ?', (name,))
            conn.commit()
            return cursor.rowcount > 0
        except Exception as e:
            print(f"Error deleting place: {e}")
            conn.rollback()
            return False
        finally:
            conn.close()
    
    def record_recognition(self, place_name: str, confidence: float):
        """Record a recognition event"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Add to history
            cursor.execute('''
                INSERT INTO recognition_history (place_name, confidence)
                VALUES (?, ?)
            ''', (place_name, confidence))
            
            # Update place statistics
            cursor.execute('''
                UPDATE places
                SET recognition_count = recognition_count + 1,
                    last_recognized = CURRENT_TIMESTAMP,
                    avg_confidence = (
                        SELECT AVG(confidence)
                        FROM recognition_history
                        WHERE place_name = ?
                    )
                WHERE name = ?
            ''', (place_name, place_name))
            
            conn.commit()
            return True
        except Exception as e:
            print(f"Error recording recognition: {e}")
            conn.rollback()
            return False
        finally:
            conn.close()
    
    def get_place_stats(self, name: str) -> Optional[Dict]:
        """Get statistics for a specific place"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                SELECT 
                    name,
                    created_at,
                    updated_at,
                    description,
                    image_count,
                    recognition_count,
                    last_recognized,
                    avg_confidence
                FROM places
                WHERE name = ?
            ''', (name,))
            
            row = cursor.fetchone()
            if row:
                return dict(row)
            return None
        finally:
            conn.close()
    
    def get_all_places_stats(self) -> List[Dict]:
        """Get statistics for all places"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                SELECT 
                    name,
                    created_at,
                    updated_at,
                    description,
                    image_count,
                    recognition_count,
                    last_recognized,
                    avg_confidence
                FROM places
                ORDER BY recognition_count DESC
            ''')
            
            return [dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()
    
    def get_recognition_history(self, limit: int = 50) -> List[Dict]:
        """Get recent recognition history"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                SELECT place_name, confidence, timestamp
                FROM recognition_history
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (limit,))
            
            return [dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()
    
    def export_data(self) -> Dict:
        """Export all data for backup"""
        return {
            'places': self.get_all_places_stats(),
            'history': self.get_recognition_history(limit=1000),
            'exported_at': datetime.now().isoformat()
        }
    
    def get_place_images(self, name: str) -> List[str]:
        """Get all image paths for a place"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                SELECT image_path
                FROM place_images
                WHERE place_name = ?
                ORDER BY added_at DESC
            ''', (name,))
            
            return [row[0] for row in cursor.fetchall()]
        finally:
            conn.close()
