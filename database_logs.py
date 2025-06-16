from sqlalchemy import create_engine, Column, Integer, DateTime, Boolean, ForeignKey
from sqlalchemy.orm import sessionmaker, Session, relationship
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

# Create a separate engine for logs
logs_engine = create_engine("postgresql://avnadmin:AVNS_dUD6pIEd7VcwBKC3nwa@irchad-irchad.e.aivencloud.com:15147/defaultdb?sslmode=require")
LogsSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=logs_engine)
LogsBase = declarative_base()

class NavigationLogs(LogsBase):
    __tablename__ = "navigation_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, default=1)  # Dummy user ID
    environment_id = Column(Integer, default=1)  # Dummy environment ID
    rerouting_count = Column(Integer, default=0)
    start_time = Column(DateTime, default=datetime.now)
    end_time = Column(DateTime, nullable=True)
    
    poiLogs = relationship("POILogs", back_populates="navigation")
    zoneLogs = relationship("ZoneLogs", back_populates="navigation")

class ZoneLogs(LogsBase):
    __tablename__ = "zone_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    navigation_id = Column(Integer, ForeignKey("navigation_logs.id"))
    zone_id = Column(Integer)  # Keep as Integer
    user_id = Column(Integer, default=1)  # Dummy user ID
    start_time = Column(DateTime, default=datetime.now)
    end_time = Column(DateTime, default=datetime.now)
    obstacles_encountered = Column(Integer, default=0)
    
    navigation = relationship("NavigationLogs", back_populates="zoneLogs")

class POILogs(LogsBase):
    __tablename__ = "poi_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    navigation_id = Column(Integer, ForeignKey("navigation_logs.id"))
    poi_id = Column(Integer)  # Keep as Integer
    visit_time = Column(DateTime, default=datetime.now)
    
    navigation = relationship("NavigationLogs", back_populates="poiLogs")

class DeviceUsageLogs(LogsBase):
    __tablename__ = "device_usage_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    dispositive_id = Column(Integer, default=1)  # Dummy device ID
    timestamp = Column(DateTime, default=datetime.now)
    battery_level = Column(Integer, default=100)  # Dummy battery level
    connected = Column(Boolean, default=True)  # Dummy connection status

# Create all tables
LogsBase.metadata.create_all(bind=logs_engine)

# Dependency to get logs database session
def get_logs_db():
    db = LogsSessionLocal()
    try:
        yield db
    finally:
        db.close() 