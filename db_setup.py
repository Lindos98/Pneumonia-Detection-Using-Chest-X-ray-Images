# db_setup.py

import sqlite3

def create_database():
    conn = sqlite3.connect('database.db')  # This will create database.db file
    cursor = conn.cursor()

    # Create users table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        email TEXT NOT NULL,
        role TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')

    # Create images table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS images (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        original_image_filename TEXT NOT NULL,
        preprocessed_image_filename TEXT,
        upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        image_hash TEXT UNIQUE,
        FOREIGN KEY (user_id) REFERENCES users(id)
    )
    ''')

    # Create predictions table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        image_id INTEGER NOT NULL,
        model_name TEXT NOT NULL,
        prediction_label TEXT NOT NULL,
        confidence_score REAL NOT NULL,
        prediction_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (image_id) REFERENCES images(id)
    )
    ''')

    conn.commit()
    conn.close()
    print("Database and tables created successfully.")

if __name__ == "__main__":
    create_database()
