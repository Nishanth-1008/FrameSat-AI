import sqlite3
import os

class GOES19MetadataDB:
    def __init__(self, db_path):
        self.db_path = db_path
        self.init_db()

    def _get_connection(self):
        return sqlite3.connect(self.db_path)

    def init_db(self):
        """Initialize the SQLite database and create the scenes table if it doesn't exist."""
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS scenes (
                    scene_id TEXT PRIMARY KEY,
                    timestamp TEXT,
                    satellite TEXT,
                    channel INTEGER,
                    sector TEXT,
                    scan_mode TEXT,
                    filepath TEXT,
                    filesize INTEGER,
                    checksum TEXT,
                    download_time TEXT
                )
            """)
            conn.commit()

    def insert_scene(self, scene_id, timestamp, satellite, channel, sector, scan_mode, filepath, filesize, checksum, download_time):
        """Insert or replace a scene record in the database."""
        with self._get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO scenes (
                    scene_id, timestamp, satellite, channel, sector, scan_mode, filepath, filesize, checksum, download_time
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (scene_id, timestamp, satellite, channel, sector, scan_mode, filepath, filesize, checksum, download_time))
            conn.commit()

    def remove_scene(self, scene_id):
        """Remove a scene record by ID."""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM scenes WHERE scene_id = ?", (scene_id,))
            conn.commit()

    def get_valid_scene_count(self, satellite, channel, sector):
        """Return the count of valid scenes matching the configuration."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT COUNT(*) FROM scenes 
                WHERE satellite = ? AND channel = ? AND sector = ?
            """, (satellite, channel, sector))
            return cursor.fetchone()[0]

    def get_scene_by_id(self, scene_id):
        """Retrieve a single scene record by ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM scenes WHERE scene_id = ?", (scene_id,))
            row = cursor.fetchone()
            if row:
                columns = [col[0] for col in cursor.description]
                return dict(zip(columns, row))
            return None

    def get_all_scenes(self, satellite=None, channel=None, sector=None, sorted_chronologically=True):
        """Retrieve all scenes, optionally filtered, and sorted by timestamp."""
        query = "SELECT * FROM scenes"
        params = []
        conditions = []
        
        if satellite:
            conditions.append("satellite = ?")
            params.append(satellite)
        if channel is not None:
            conditions.append("channel = ?")
            params.append(channel)
        if sector:
            conditions.append("sector = ?")
            params.append(sector)
            
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
            
        if sorted_chronologically:
            query += " ORDER BY timestamp ASC"
            
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
            columns = [col[0] for col in cursor.description]
            return [dict(zip(columns, row)) for row in rows]
