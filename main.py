# main.py
import os
import cv2
import numpy as np
import base64
import tempfile
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from inference_sdk import InferenceHTTPClient, InferenceConfiguration
import io
import matplotlib.pyplot as plt

# Initialize FastAPI app
app = FastAPI(
    title="Floor Plan Processing API",
    description="API for detecting walls, doors, and furniture in floor plan images and converting to grid format",
    version="1.0.0"
)

# Add CORS middleware to allow cross-origin requests from your frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Roboflow client with API key - you might want to use environment variables for this
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="cU8I30jSejfvnYUYqvnk"  # Better to use environment variables
)

# Set custom inference configurations for different models
text_config = InferenceConfiguration(confidence_threshold=0.45)
walls_doors_config = InferenceConfiguration(confidence_threshold=0.4)
furniture_config = InferenceConfiguration(confidence_threshold=0.1)

# Define request models
class ProcessOptions(BaseModel):
    grid_size: int = 4  # Default grid size
    include_text_removal: bool = True
    include_walls_detection: bool = True
    include_furniture_detection: bool = True
    include_doors_detection: bool = False  # New parameter

# ==================== TEXT DETECTION AND REMOVAL ====================

def get_text_bboxes(image_path, rotation=0):
    """
    Runs OCR inference on an image and returns text bounding boxes.
    Handles rotated images and transforms coordinates back to original orientation.

    Args:
        image_path: Path to the image
        rotation: Rotation angle in degrees (0, 90, 180, 270)

    Returns:
        List of bounding boxes in original image coordinates
    """
    # Load the original image to get dimensions
    orig_img = cv2.imread(image_path)
    if orig_img is None:
        raise ValueError(f"Failed to load image at {image_path}")

    orig_height, orig_width = orig_img.shape[:2]

    # If rotation is needed, create a temporary rotated image
    if rotation != 0:
        # Rotate the image
        if rotation == 90:
            rotated_img = cv2.rotate(orig_img, cv2.ROTATE_90_CLOCKWISE)
        elif rotation == 180:
            rotated_img = cv2.rotate(orig_img, cv2.ROTATE_180)
        elif rotation == 270:
            rotated_img = cv2.rotate(orig_img, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # Save the rotated image to a temporary file
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
            temp_path = temp_file.name

        cv2.imwrite(temp_path, rotated_img)
        inference_path = temp_path

        # Get dimensions of rotated image
        rot_height, rot_width = rotated_img.shape[:2]
    else:
        inference_path = image_path
        rot_height, rot_width = orig_height, orig_width

    # Run inference on the image
    with CLIENT.use_configuration(text_config):
        result = CLIENT.infer(inference_path, model_id="ocr-alphademo/1")

    # Clean up temporary file if created
    if rotation != 0:
        os.remove(temp_path)

    # Process predictions and convert coordinates back to original orientation
    bboxes = []
    for prediction in result["predictions"]:
        x, y, w, h = prediction["x"], prediction["y"], prediction["width"], prediction["height"]
        confidence = prediction.get("confidence", 0)
        text = prediction.get("class", "")

        # Calculate corners in rotated image
        x1_rot, y1_rot = int(x - w / 2), int(y - h / 2)
        x2_rot, y2_rot = int(x + w / 2), int(y + h / 2)

        # Transform coordinates back to original orientation
        if rotation == 0:
            x1, y1, x2, y2 = x1_rot, y1_rot, x2_rot, y2_rot
        elif rotation == 90:
            # For 90° clockwise, (x, y) in rotated becomes (y, width-x) in original
            x1 = y1_rot
            y1 = rot_width - x2_rot
            x2 = y2_rot
            y2 = rot_width - x1_rot
        elif rotation == 180:
            # For 180°, (x, y) in rotated becomes (width-x, height-y) in original
            x1 = rot_width - x2_rot
            y1 = rot_height - y2_rot
            x2 = rot_width - x1_rot
            y2 = rot_height - y1_rot
        elif rotation == 270:
            # For 270° clockwise, (x, y) in rotated becomes (height-y, x) in original
            x1 = rot_height - y2_rot
            y1 = x1_rot
            x2 = rot_height - y1_rot
            y2 = x2_rot

        # Ensure coordinates are within image boundaries
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(orig_width, x2), min(orig_height, y2)

        # Skip invalid boxes (could happen due to rounding errors)
        if x1 >= x2 or y1 >= y2:
            continue

        bboxes.append((x1, y1, x2, y2, confidence, text, rotation))

    return bboxes

def get_all_text_bboxes(image_path):
    """
    Gets text bounding boxes by running detection in all four orientations.
    """
    all_bboxes = []

    for rotation in [0, 90, 180, 270]:
        bboxes = get_text_bboxes(image_path, rotation)
        all_bboxes.extend(bboxes)

    return all_bboxes

def remove_overlapping_boxes(bboxes, iou_threshold=0.5):
    """
    Remove overlapping bounding boxes with lower confidence scores.

    Args:
        bboxes: List of bounding boxes in format (x1, y1, x2, y2, confidence, text, rotation)
        iou_threshold: Intersection over Union threshold

    Returns:
        Filtered list of bounding boxes
    """
    # Sort boxes by confidence score (higher first)
    sorted_boxes = sorted(bboxes, key=lambda x: x[4], reverse=True)

    # List to store kept boxes
    kept_boxes = []

    for box in sorted_boxes:
        x1a, y1a, x2a, y2a = box[0], box[1], box[2], box[3]
        area_a = (x2a - x1a) * (y2a - y1a)

        # Flag to determine if current box overlaps with any kept box
        should_keep = True

        for kept_box in kept_boxes:
            x1b, y1b, x2b, y2b = kept_box[0], kept_box[1], kept_box[2], kept_box[3]

            # Calculate intersection area
            x1i = max(x1a, x1b)
            y1i = max(y1a, y1b)
            x2i = min(x2a, x2b)
            y2i = min(y2a, y2b)

            if x1i < x2i and y1i < y2i:  # If there is intersection
                area_i = (x2i - x1i) * (y2i - y1i)
                area_b = (x2b - x1b) * (y2b - y1b)

                # Calculate IoU
                iou = area_i / (area_a + area_b - area_i)

                if iou > iou_threshold:
                    should_keep = False
                    break

        if should_keep:
            kept_boxes.append(box)

    return kept_boxes

def create_text_mask(image_shape, bboxes, padding=2):
    """
    Creates a binary mask from text bounding boxes with optional padding.
    """
    mask = np.zeros(image_shape[:2], dtype=np.uint8)

    for (x1, y1, x2, y2, _, _, _) in bboxes:
        # Add padding to ensure we capture the full text
        x1_pad = max(0, x1 - padding)
        y1_pad = max(0, y1 - padding)
        x2_pad = min(image_shape[1], x2 + padding)
        y2_pad = min(image_shape[0], y2 + padding)

        # Draw filled rectangle on mask
        cv2.rectangle(mask, (x1_pad, y1_pad), (x2_pad, y2_pad), 255, thickness=cv2.FILLED)

    return mask

def remove_text(image, text_mask, method='inpaint'):
    """
    Removes text from an image based on the text mask.
    """
    if method == 'inpaint':
        # Use inpainting to fill in the text areas
        return cv2.inpaint(image, text_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    elif method == 'fill':
        # Simple white fill
        image_copy = image.copy()
        image_copy[text_mask == 255] = (255, 255, 255)
        return image_copy
    else:
        raise ValueError("Invalid method. Choose 'inpaint' or 'fill'")

def process_text_removal(image_path):
    """
    Process a floor plan image to detect and remove text in all orientations.
    """
    # Load floor plan
    original_image = cv2.imread(image_path)
    if original_image is None:
        raise ValueError(f"Failed to load image at {image_path}")

    # Get text bounding boxes from all orientations
    all_text_bboxes = get_all_text_bboxes(image_path)

    # Remove overlapping detections
    filtered_text_bboxes = remove_overlapping_boxes(all_text_bboxes)

    # Create mask for text regions
    text_mask = create_text_mask(original_image.shape, filtered_text_bboxes)

    # Remove text from the image
    no_text_image = remove_text(original_image, text_mask, method='inpaint')

    return original_image, no_text_image, filtered_text_bboxes, text_mask

# ==================== WALLS AND DOORS DETECTION ====================

def get_walls_doors_bboxes(image_path):
    """
    Gets bounding boxes for walls and doors from respective models.
    Returns them separately.
    """
    # Get doors bounding boxes
    with CLIENT.use_configuration(walls_doors_config):
        doors_result = CLIENT.infer(image_path, model_id="doors-zkxtc/1")

    doors_bboxes = []
    for prediction in doors_result["predictions"]:
        x, y, w, h = prediction["x"], prediction["y"], prediction["width"], prediction["height"]
        confidence = prediction.get("confidence", 0)
        x1, y1 = int(x - w / 2), int(y - h / 2)
        x2, y2 = int(x + w / 2), int(y + h / 2)
        doors_bboxes.append((x1, y1, x2, y2, "door", confidence))

    # Get walls bounding boxes
    with CLIENT.use_configuration(walls_doors_config):
        walls_result = CLIENT.infer(image_path, model_id="sketch-detection-walls/1")

    walls_bboxes = []
    for prediction in walls_result["predictions"]:
        x, y, w, h = prediction["x"], prediction["y"], prediction["width"], prediction["height"]
        confidence = prediction.get("confidence", 0)
        # Thicken walls slightly for better visualization
        if w > h:
            h *= 1.3  # Thicken horizontal walls
        else:
            w *= 1.3  # Thicken vertical walls
        x1, y1 = int(x - w / 2), int(y - h / 2)
        x2, y2 = int(x + w / 2), int(y + h / 2)
        walls_bboxes.append((x1, y1, x2, y2, "wall", confidence))

    return walls_bboxes, doors_bboxes

def create_mask_from_bboxes(image_shape, bboxes):
    """
    Creates a binary mask from bounding boxes
    """
    mask = np.zeros(image_shape[:2], dtype=np.uint8)

    for (x1, y1, x2, y2, _, _) in bboxes:
        # Draw filled rectangle on mask
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, thickness=cv2.FILLED)

    return mask

def extract_elements(image, mask):
    """
    Extracts elements based on a binary mask
    """
    result = np.ones_like(image) * 255  # White background
    result[mask == 255] = image[mask == 255]  # Copy only the masked regions
    return result

def remove_elements(image, mask, method='inpaint'):
    """
    Removes elements from an image based on a mask.
    """
    if method == 'inpaint':
        return cv2.inpaint(image, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    elif method == 'fill':
        image_copy = image.copy()
        image_copy[mask == 255] = (255, 255, 255)  # Fill with white
        return image_copy
    else:
        raise ValueError("Invalid method. Choose 'inpaint' or 'fill'")

def process_walls_doors(image_path, no_text_image):
    """
    Process a floor plan image to detect walls and doors.
    Returns images with only walls, only doors, and without walls/doors separately.
    """
    # Save no_text_image to disk temporarily for model inference
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
        temp_no_text_path = temp_file.name
    
    cv2.imwrite(temp_no_text_path, no_text_image)

    # Get bounding boxes for walls and doors separately
    walls_bboxes, doors_bboxes = get_walls_doors_bboxes(temp_no_text_path)

    # Create separate masks
    walls_mask = create_mask_from_bboxes(no_text_image.shape, walls_bboxes)
    doors_mask = create_mask_from_bboxes(no_text_image.shape, doors_bboxes)
    combined_mask = cv2.bitwise_or(walls_mask, doors_mask)

    # Extract walls only
    walls_only = extract_elements(no_text_image, walls_mask)
    
    # Extract doors only
    doors_only = extract_elements(no_text_image, doors_mask)

    # Create image without walls and doors (for furniture detection)
    no_walls_doors = remove_elements(no_text_image, combined_mask, method='fill')
    
    # Create image without walls but with doors (for modified pipeline)
    no_walls_with_doors = remove_elements(no_text_image, walls_mask, method='fill')

    # Clean up
    os.remove(temp_no_text_path)

    return walls_only, doors_only, no_walls_doors, no_walls_with_doors, walls_bboxes, doors_bboxes, walls_mask, doors_mask, combined_mask

# ==================== FURNITURE DETECTION ====================

def get_furniture_bboxes(image_path):
    """
    Gets bounding boxes for furniture from the furniture detection model.
    """
    with CLIENT.use_configuration(furniture_config):
        result = CLIENT.infer(image_path, model_id="floorplan-object-detection5.0/1")

    furniture_bboxes = []
    for prediction in result["predictions"]:
        x, y, w, h = prediction["x"], prediction["y"], prediction["width"], prediction["height"]
        confidence = prediction.get("confidence", 0)
        class_name = prediction.get("class", "furniture")
        x1, y1 = int(x - w / 2), int(y - h / 2)
        x2, y2 = int(x + w / 2), int(y + h / 2)
        furniture_bboxes.append((x1, y1, x2, y2, class_name, confidence))

    return furniture_bboxes

def process_furniture(no_walls_doors_image):
    """
    Process the no-walls-doors image to detect furniture.
    Returns images with only furniture and without furniture, with furniture represented as black masks.
    """
    # Save image temporarily for model inference
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
        temp_path = temp_file.name
    
    cv2.imwrite(temp_path, no_walls_doors_image)

    # Get furniture bounding boxes
    furniture_bboxes = get_furniture_bboxes(temp_path)

    # Create mask for furniture - BLACK filled shapes with reduced size (0.8 scale)
    furniture_mask = np.zeros(no_walls_doors_image.shape[:2], dtype=np.uint8)

    for (x1, y1, x2, y2, _, _) in furniture_bboxes:
        # Calculate center of the bounding box
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        # Calculate width and height
        width = x2 - x1
        height = y2 - y1

        # Scale down by 0.8
        new_width = int(width * 0.8)
        new_height = int(height * 0.8)

        # Calculate new coordinates around the same center
        new_x1 = center_x - new_width // 2
        new_y1 = center_y - new_height // 2
        new_x2 = center_x + new_width // 2
        new_y2 = center_y + new_height // 2

        # Draw scaled rectangle on mask
        cv2.rectangle(furniture_mask, (new_x1, new_y1), (new_x2, new_y2), 255, thickness=cv2.FILLED)

    # Create black-filled furniture image
    black_furniture = np.ones_like(no_walls_doors_image) * 255  # White background
    black_furniture[furniture_mask == 255] = 0  # Black furniture

    # Create image without furniture
    no_furniture = remove_elements(no_walls_doors_image, furniture_mask, method='fill')

    # Clean up
    os.remove(temp_path)

    return black_furniture, furniture_bboxes, furniture_mask

# ==================== FINAL INTEGRATION ====================

def combine_final_result(original_image, no_text_image, walls_doors_mask, furniture_mask, grid_size=1):
    """
    Creates a final binary image with walls, doors, and furniture as black (1)
    and empty spaces as white (0), divided into a grid of small squares.

    Args:
        original_image: Original floor plan image
        no_text_image: Image with text removed
        walls_doors_mask: Binary mask for walls and doors
        furniture_mask: Binary mask for furniture
        grid_size: Size of each grid cell in pixels (smaller = higher resolution)
    """
    height, width = original_image.shape[:2]

    # Calculate grid dimensions
    grid_rows = height // grid_size
    grid_cols = width // grid_size

    # Create a combined binary mask for walls, doors, and furniture
    combined_mask = cv2.bitwise_or(walls_doors_mask, furniture_mask)

    # Create an empty grid (0 = free space, 1 = obstacle)
    grid = np.zeros((grid_rows, grid_cols), dtype=np.uint8)

    # For each grid cell, check if it contains any obstacle
    for row in range(grid_rows):
        for col in range(grid_cols):
            # Get the corresponding area in the mask
            top = row * grid_size
            left = col * grid_size
            bottom = min((row + 1) * grid_size, height)
            right = min((col + 1) * grid_size, width)

            # Extract the region from the mask
            region = combined_mask[top:bottom, left:right]

            # If any pixel in this region is an obstacle (255), mark the cell as obstacle (1)
            if np.any(region > 0):
                grid[row, col] = 1

    # Create a visual representation of the grid
    grid_visual = np.ones((height, width), dtype=np.uint8) * 255

    # Fill grid cells with black if they contain obstacles
    for row in range(grid_rows):
        for col in range(grid_cols):
            if grid[row, col] == 1:
                top = row * grid_size
                left = col * grid_size
                bottom = min((row + 1) * grid_size, height)
                right = min((col + 1) * grid_size, width)

                grid_visual[top:bottom, left:right] = 0

    # Create grid with lines for visualization
    grid_with_lines = grid_visual.copy()

    # Draw horizontal grid lines
    for row in range(grid_rows + 1):
        y = min(row * grid_size, height - 1)
        cv2.line(grid_with_lines, (0, y), (width, y), 128, 1)

    # Draw vertical grid lines
    for col in range(grid_cols + 1):
        x = min(col * grid_size, width - 1)
        cv2.line(grid_with_lines, (x, 0), (x, height), 128, 1)

    return grid_visual, grid, grid_with_lines

def process_floor_plan(image_path, grid_size=4, include_text_removal=True, include_walls_detection=True, include_furniture_detection=True, include_doors_detection=False):
    """
    Complete pipeline for floor plan processing with option to exclude doors:
    1. Remove text
    2. Detect walls (optionally detect doors)
    3. Detect furniture as black masks (reduced size)
    4. Create final binary grid with obstacles (1) and free space (0)

    Args:
        image_path: Path to the floor plan image
        grid_size: Size of grid cells in pixels
        include_text_removal: Whether to include text removal step
        include_walls_detection: Whether to include walls detection step
        include_furniture_detection: Whether to include furniture detection step
        include_doors_detection: Whether to include doors in the detection (new parameter)

    Returns:
        Dictionary with all processed images and detected elements
    """
    # Step 1: Text Detection and Removal
    if include_text_removal:
        original_image, no_text_image, text_bboxes, text_mask = process_text_removal(image_path)
    else:
        original_image = cv2.imread(image_path)
        no_text_image = original_image.copy()
        text_bboxes = []
        text_mask = np.zeros(original_image.shape[:2], dtype=np.uint8)

    # Step 2: Walls and Doors Detection
    if include_walls_detection:
        walls_only, doors_only, no_walls_doors, no_walls_with_doors, walls_bboxes, doors_bboxes, walls_mask, doors_mask, combined_mask = process_walls_doors(image_path, no_text_image)
    else:
        walls_only = np.ones_like(original_image) * 255
        doors_only = np.ones_like(original_image) * 255
        no_walls_doors = no_text_image.copy()
        no_walls_with_doors = no_text_image.copy()
        walls_bboxes = []
        doors_bboxes = []
        walls_mask = np.zeros(original_image.shape[:2], dtype=np.uint8)
        doors_mask = np.zeros(original_image.shape[:2], dtype=np.uint8)
        combined_mask = np.zeros(original_image.shape[:2], dtype=np.uint8)

    # Step 3: Furniture Detection as Black Masks (reduced size)
    # Choose the correct input image based on whether doors should be included
    furniture_input_image = no_walls_doors if not include_doors_detection else no_walls_with_doors
    
    if include_furniture_detection:
        black_furniture, furniture_bboxes, furniture_mask = process_furniture(furniture_input_image)
    else:
        black_furniture = np.ones_like(original_image) * 255
        furniture_bboxes = []
        furniture_mask = np.zeros(original_image.shape[:2], dtype=np.uint8)

    # Step 4: Final Integration as Small-celled Binary Grid
    # Choose the appropriate mask for doors based on the include_doors_detection parameter
    doors_mask_for_grid = doors_mask if include_doors_detection else np.zeros_like(doors_mask)
    
    # Create the final mask combining walls and furniture (and doors if included)
    final_walls_mask = walls_mask
    final_obstacles_mask = cv2.bitwise_or(final_walls_mask, doors_mask_for_grid)
    final_mask = cv2.bitwise_or(final_obstacles_mask, furniture_mask)
    
    grid_visual, grid, grid_with_lines = combine_final_result(
        original_image, no_text_image, final_obstacles_mask, furniture_mask, grid_size=grid_size)

    # Get grid dimensions
    grid_rows, grid_cols = grid.shape

    # Return all processed images and detected elements
    return {
        "original_image": original_image,
        "no_text_image": no_text_image,
        "walls_only": walls_only,
        "doors_only": doors_only,
        "no_walls_doors": no_walls_doors,
        "no_walls_with_doors": no_walls_with_doors,
        "black_furniture": black_furniture,
        "grid_visual": grid_visual,
        "grid_with_lines": grid_with_lines,
        "grid": grid,
        "grid_dimensions": (grid_rows, grid_cols),
        "grid_size": grid_size,
        "text_bboxes": text_bboxes,
        "walls_bboxes": walls_bboxes,
        "doors_bboxes": doors_bboxes,
        "furniture_bboxes": furniture_bboxes
    }


def encode_image_to_base64(image):
    """Convert a numpy image to base64 encoded string."""
    if image is None:
        return None
    success, encoded_image = cv2.imencode('.png', image)
    if not success:
        return None
    return base64.b64encode(encoded_image).decode('utf-8')

# API endpoints
@app.post("/process_floor_plan")
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

@app.get("/")
async def root():
    """API status endpoint."""
    return {"status": "Floor Plan Processing API is running"}

# Run the server with uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)