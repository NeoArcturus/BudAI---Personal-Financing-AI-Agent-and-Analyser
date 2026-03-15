from flask import Blueprint, jsonify, request
import traceback
from middleware.auth_middleware import token_required
from services.api_integrator.get_account_detail import UserAccounts

account_bp = Blueprint('accounts', __name__)


@account_bp.route('/', methods=['GET'])
@token_required
def get_accounts():
    try:
        user_acc = UserAccounts(user_id=request.user_uuid)
        all_accounts = user_acc.get_all_accounts()
        return jsonify({"accounts": all_accounts})
    except Exception:
        traceback.print_exc()
        return jsonify({"accounts": []}), 500


@account_bp.route('/<account_id>/transactions', methods=['GET'])
@token_required
def get_transactions(account_id):
    try:
        user_acc = UserAccounts(user_id=request.user_uuid)
        transactions = user_acc.get_transactions_by_account(account_id)
        return jsonify({"transactions": transactions})
    except Exception:
        traceback.print_exc()
        return jsonify({"transactions": []}), 500


@account_bp.route('/<provider_id>', methods=['DELETE'])
@token_required
def revoke_connection(provider_id):
    try:
        user_acc = UserAccounts(user_id=request.user_uuid)
        success = user_acc.revoke_provider_connection(provider_id)
        if success:
            return jsonify({"status": "success"})
        return jsonify({"error": "Failed to revoke connection"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500
