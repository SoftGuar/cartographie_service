from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.endpoints.floor_plan_processing import router as floor_plan_router
from api.endpoints.floors import router as floor_router
from api.endpoints.environments import router as environment_router
from database import Base, engine

Base.metadata.create_all(bind=engine)

# Create the FastAPI app
app = FastAPI(
    title="Floor Plan Processing API",
    description="API for detecting walls, doors, and furniture in floor plan images and converting to grid format",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the floor plan processing router
app.include_router(floor_plan_router)
app.include_router(floor_router)
app.include_router(environment_router)

@app.get("/")
async def root():
    """API status endpoint."""
    return {"status": "Floor Plan Processing API is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)