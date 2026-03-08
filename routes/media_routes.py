from flask import Blueprint, jsonify, Response
import os
import pandas as pd
from middleware.auth_middleware import token_required

media_bp = Blueprint('media', __name__)


@media_bp.route('/csv/<filename>', methods=['GET'])
@token_required
def serve_csv(filename):
    csv_dir = os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..', 'saved_media', 'csvs'))
    file_path = os.path.join(csv_dir, filename)

    try:
        if "converged" in filename or "hybrid" in filename:
            with open(file_path, 'r') as f:
                return Response(f.read(), mimetype='text/csv')

        df = pd.read_csv(file_path)
        return jsonify({"data": df.fillna("").to_dict(orient="records")})
    except Exception:
        return jsonify({"data": []}), 404
