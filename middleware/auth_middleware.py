from functools import wraps
from flask import request, jsonify
import sqlite3


def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({'message': 'Token is missing'}), 401

        token_str = token.split(" ")[1] if "Bearer " in token else token

        try:
            with sqlite3.connect("budai_memory.db") as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT user_uuid, name FROM users WHERE user_uuid = ?", (token_str,))
                user = cursor.fetchone()

            if not user:
                return jsonify({'message': 'Token is invalid'}), 401

            request.user_uuid = user[0]
            request.user_name = user[1]

        except Exception as e:
            return jsonify({'message': str(e)}), 500

        return f(*args, **kwargs)
    return decorated
