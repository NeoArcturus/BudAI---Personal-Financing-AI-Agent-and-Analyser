from flask import Blueprint, request, jsonify, redirect
from services.api_integrator.access_token_generator import AccessTokenGenerator
from services.user_service import UserService
from middleware.auth_middleware import token_required

auth_bp = Blueprint('auth', __name__)
callback_bp = Blueprint('callback', __name__)


@auth_bp.route('/login', methods=['POST'])
def login():
    data = request.json
    email = data.get("email")
    password = data.get("password")

    user_service = UserService()
    user_uuid = user_service.authenticate_or_create_user(email, password)

    return jsonify({"token": user_uuid, "status": "success"})


@auth_bp.route('/truelayer/status', methods=['GET'])
@token_required
def truelayer_status():
    token_gen = AccessTokenGenerator()
    return jsonify({"auth_url": token_gen.get_auth_link(request.user_uuid)})


@callback_bp.route('/callback', methods=['GET'])
def truelayer_callback():
    code = request.args.get('code')
    state = request.args.get('state')

    token_gen = AccessTokenGenerator()
    if token_gen.validate_callback(code, state):
        return redirect("http://localhost:3000/dashboard")

    return jsonify({"error": "Authentication failed or session expired"}), 400


@auth_bp.route('/connections/extend', methods=['POST'])
@token_required
def extend_user_connections():
    data = request.json
    provider_ids = data.get('provider_ids', [])

    token_gen = AccessTokenGenerator()
    results = token_gen.extend_providers(provider_ids, request.user_uuid)

    return jsonify({"results": results})


@auth_bp.route('/connections/revoke', methods=['POST'])
@token_required
def revoke_truelayer_access():
    data = request.json or {}
    provider_id = data.get("provider_id")

    token_gen = AccessTokenGenerator()
    results = token_gen.revoke_provider(provider_id, request.user_uuid)

    if results and results[0].get("status") == "not_found":
        return jsonify({"error": results[0].get("error")}), 404

    return jsonify({"results": results})
