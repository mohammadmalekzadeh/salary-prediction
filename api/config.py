from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base

DATABASE_URL = 'sqlite:///api/database/salary.db'
Base = declarative_base()
engine = create_engine(DATABASE_URL)
Base.metadata.create_all(engine)