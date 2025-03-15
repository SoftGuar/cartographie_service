from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
import os

# Create the database directory if it doesn't exist
os.makedirs('data', exist_ok=True)

# Use SQLite with a file in the data directory
SQLALCHEMY_DATABASE_URL = "sqlite:///data/indoor_mapping.db"

# Create engine with SQLite support for concurrent access
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False}  # Needed for SQLite
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()