from config import Base, engine


def init_db(db_path=None):
    Base.metadata.create_all(bind=engine)
