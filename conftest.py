import os
import sys
from pathlib import Path

# Add the project root directory to Python path
project_root = str(Path(__file__).parent)
sys.path.append(project_root)

# Import after path is set
from main import app
from database import Base, engine

# Create tables
Base.metadata.create_all(bind=engine)