import sqlite3
import uuid


class UserService:
    def __init__(self, db_path="budai_memory.db"):
        self.db_path = db_path

    def authenticate_or_create_user(self, email, password):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT user_uuid FROM users WHERE name = ? AND password = ?", (email, password))
            user = cursor.fetchone()

            if user:
                return user[0]

            new_uuid = str(uuid.uuid4())
            conn.execute(
                "INSERT INTO users (user_uuid, name, password) VALUES (?, ?, ?)", (new_uuid, email, password))
            return new_uuid
