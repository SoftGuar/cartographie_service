import cv2
import numpy as np
import tempfile
import os
from . import CLIENT, walls_doors_config

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
    Creates a binary mask from bounding boxes.
    """
    mask = np.zeros(image_shape[:2], dtype=np.uint8)

    for (x1, y1, x2, y2, _, _) in bboxes:
        # Draw filled rectangle on mask
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, thickness=cv2.FILLED)

    return mask

def extract_elements(image, mask):
    """
    Extracts elements based on a binary mask.
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