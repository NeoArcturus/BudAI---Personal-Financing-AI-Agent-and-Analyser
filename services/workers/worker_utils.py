from typing import Any, Mapping

def resolve_user_uuid(state: Mapping[str, Any]) -> str:
    """
    Safely resolves the active user UUID from agent state or the database.
    Eliminates hardcoded phantom UUIDs across all multi-agent worker tools.
    """
    user_uuid = state.get("user_uuid") or state.get("user_id")
    if user_uuid and str(user_uuid).strip().lower() not in ["", "none", "null", "undefined"]:
        return str(user_uuid)

    from config import SessionLocal
    from models.database_models import User
    with SessionLocal() as session:
        u = session.query(User.user_uuid).first()
        if u and u[0]:
            return str(u[0])

    raise ValueError("Missing 'user_uuid' in agent state and no registered user found in the database.")
