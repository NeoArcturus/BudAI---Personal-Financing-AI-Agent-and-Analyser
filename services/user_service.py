import uuid
from config import SessionLocal
from models.database_models import User


class UserService:
    def __init__(self, db_path=None):
        pass

    def authenticate_or_create_user(self, email, password):
        with SessionLocal() as session:
            user = session.query(User).filter(
                User.name == email, User.password == password).first()

            if user:
                return user.user_uuid

            new_uuid = str(uuid.uuid4())
            new_user = User(user_uuid=new_uuid, name=email, password=password)
            session.add(new_user)
            session.commit()
            return new_uuid
