import base64
import cv2
import numpy as np
from detection.text_detection import process_text_removal
from detection.walls_doors_detection import process_walls_doors
from detection.furniture_detection import process_furniture
from detection.grid_processing import combine_final_result

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