from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, String, Integer, Float
from config import engine

Base = declarative_base()

class SalaryStatistics(Base):
    __tablename__ = 'salary_statistics'
    id = Column(Integer, primary_key=True, autoincrement=True)
    category = Column(String)
    value = Column(String)
    mean_salary = Column(Float)

class OrdinalEncoding(Base):
    __tablename__ = 'ordinal_encodings'
    id = Column(Integer, primary_key=True, autoincrement=True)
    category = Column(String)
    value = Column(String)
    ordinal_code = Column(Integer)

Base.metadata.create_all(engine)