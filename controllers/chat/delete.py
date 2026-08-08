from fastapi import HTTPException
from models.database_models import User, ChatSession, ChatHistory
from config import SessionLocal

async def delete_chat_session(session_id: str, current_user: User):
    """
    Deletes a specific chat session and all its associated message history.
    
    Args:
        session_id (str): The UUID of the chat session to delete.
        current_user (User): The authenticated user requesting the deletion.
        
    Returns:
        dict: A success status message indicating deletion or soft-deletion.
        
    Raises:
        HTTPException (500): If the database operation fails.
    """
    try:
        with SessionLocal() as session:
            chat_session = session.query(ChatSession).filter_by(
                session_id=session_id, user_uuid=current_user.user_uuid).first()
            if not chat_session:
                return {"status": "success", "message": "Session not found but considered deleted"}
            session.query(ChatHistory).filter_by(
                session_id=session_id).delete()
            session.delete(chat_session)
            session.commit()
            return {"status": "success", "message": "Session deleted"}
    except Exception:
        raise HTTPException(
            status_code=500, detail="Failed to delete session.")
