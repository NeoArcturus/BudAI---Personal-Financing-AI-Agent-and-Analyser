from flask import Blueprint, request, Response, stream_with_context
from middleware.auth_middleware import token_required
from services.budai_chat_service import execute_chat_stream

chat_bp = Blueprint('chat', __name__)


@chat_bp.route('/', methods=['POST'])
@token_required
def chat():
    data = request.json
    user_input = data.get("input")
    account_id = data.get("active_account_id")
    user_uuid = data.get("user_id")

    q = execute_chat_stream(user_input, user_uuid,
                            request.user_name, account_id)

    def generate():
        while True:
            token = q.get()
            if token is None:
                break
            yield token

    return Response(stream_with_context(generate()), mimetype='text/plain')
