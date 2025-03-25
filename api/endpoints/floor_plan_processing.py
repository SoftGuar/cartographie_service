from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional
import tempfile
import os
from services.floor_plan_service import process_floor_plan, encode_image_to_base64
from pydantic import BaseModel

router = APIRouter(prefix="/process_floor_plan", tags=["Floor Plan Processing"])

class ProcessOptions(BaseModel):
    grid_size: int = 4  # Default grid size
    include_text_removal: bool = True
    include_walls_detection: bool = True
    include_furniture_detection: bool = True
    include_doors_detection: bool = False  # New parameter

@router.post(
    "/",
    summary="Process a floor plan image",
    description="""
    Upload and process a floor plan image to detect walls, doors, and furniture.
    
    The processing pipeline includes:
    1. Text removal (optional)
    2. Wall detection
    3. Furniture detection
    4. Door detection
    5. Grid conversion
    
    The response includes:
    - Processed grid data
    - Grid dimensions
    - Intermediate processing results (if requested)
    """,
    responses={
        200: {
            "description": "Successfully processed floor plan",
            "content": {
                "application/json": {
                    "example": {
                        "grid": [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
                        "grid_dimensions": [3, 3],
                        "grid_size": 4,
                        "original_image": "base64_encoded_image",
                        "no_text_image": "base64_encoded_image",
                        "walls_only": "base64_encoded_image",
                        "black_furniture": "base64_encoded_image",
                        "grid_visual": "base64_encoded_image"
                    }
                }
            }
        },
        400: {
            "description": "Invalid file or options",
            "content": {
                "application/json": {
                    "example": {"detail": "Invalid file format. Only PNG and JPG are supported."}
                }
            }
        },
        500: {
            "description": "Processing error",
            "content": {
                "application/json": {
                    "example": {"detail": "Error processing floor plan"}
                }
            }
        }
    }
)
async def api_process_floor_plan(file: UploadFile = File(...), options: Optional[str] = Form(None)):
    """
    Process a floor plan image and return the results.
    """
    try:
        # Parse options
        if options:
            import json
            options_dict = json.loads(options)
            process_options = ProcessOptions(**options_dict)
        else:
            process_options = ProcessOptions()
        
        # Save uploaded file to temp location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
            temp_path = temp_file.name
            contents = await file.read()
            temp_file.write(contents)
        
        # Process the floor plan
        results = process_floor_plan(
            temp_path, 
            grid_size=process_options.grid_size,
            include_text_removal=process_options.include_text_removal,
            include_walls_detection=process_options.include_walls_detection,
            include_furniture_detection=process_options.include_furniture_detection,
            include_doors_detection=False  # Always set to False based on your requirements
        )
        
        # Encode images as base64 strings
        encoded_results = {
            "original_image": encode_image_to_base64(results["original_image"]),
            "no_text_image": encode_image_to_base64(results["no_text_image"]),
            "walls_only": encode_image_to_base64(results["walls_only"]),
            "no_walls_doors": encode_image_to_base64(results["no_walls_doors"]),
            "black_furniture": encode_image_to_base64(results["black_furniture"]),
            "grid_visual": encode_image_to_base64(results["grid_visual"]),
            "grid_with_lines": encode_image_to_base64(results["grid_with_lines"]),
            "grid": results["grid"].tolist(),  # Convert numpy array to list for JSON serialization
            "grid_dimensions": results["grid_dimensions"],
            "grid_size": results["grid_size"],
        }

        # Clean up temp file
        os.remove(temp_path)
        
        return JSONResponse(content=encoded_results)
    
    except Exception as e:
        # Clean up temp file if it exists
        if 'temp_path' in locals():
            os.remove(temp_path)
        
        raise HTTPException(status_code=500, detail=f"Error processing floor plan: {str(e)}")