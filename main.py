# main.py
import time
import socket
import logging
from logging.handlers import RotatingFileHandler

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pythonjsonlogger import jsonlogger

from api.endpoints.floor_plan_processing import router as floor_plan_router
from api.endpoints.floors import router as floor_router
from api.endpoints.environments import router as environment_router
from api.endpoints.zones import router as zone_router
from api.endpoints.pois import router as poi_router
from database import Base, engine

from models.zone_type import ZoneType
from models.zone import Zone
from models.floor import Floor
from models.environment import Environment
from models.poi import POI
from models.category import Category
from models.point import Point
from models.association_tables import poi_zone_association

# create tables
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Floor Plan Processing API",
    description="API for detecting walls, doors, and furniture in floor plan images and converting to grid format",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


logger = logging.getLogger("access")
logger.setLevel(logging.INFO)
handler = RotatingFileHandler("logs/access.log", maxBytes=10*1024*1024, backupCount=5)
formatter = jsonlogger.JsonFormatter(
    fmt='{"level":"%(levelname)s","time":"%(asctime)s","pid":%(process)d,'
        '"hostname":"%(hostname)s","method":"%(method)s","url":"%(url)s",'
        '"ip":"%(ip)s","headers":%(headers)s,"params":%(params)s,'
        '"query":%(query)s,"statusCode":%(status_code)d,'
        '"responseTime":"%(response_time)s","msg":"%(message)s"}',
    rename_fields={
        'levelname': 'level',
        'asctime': 'time',
        'status_code': 'statusCode'
    }
)
formatter.defaults = {"hostname": socket.gethostname()}
handler.setFormatter(formatter)
logger.addHandler(handler)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_ns = time.time_ns()
    response: Response = await call_next(request)
    duration_ms = (time.time_ns() - start_ns) / 1_000_000

    logger.info(
        "Request completed",
        extra={
            "method": request.method,
            "url": request.url.path,
            "ip": request.client.host,
            "headers": dict(request.headers),
            "params": request.path_params,
            "query": dict(request.query_params),
            "status_code": response.status_code,
            "response_time": f"{duration_ms:.2f}ms",
        }
    )
    return response

# include routers
app.include_router(floor_plan_router)
app.include_router(floor_router)
app.include_router(environment_router)
app.include_router(zone_router)
app.include_router(poi_router)

@app.get("/")
async def root():
    return {"status": "Floor Plan Processing API is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
