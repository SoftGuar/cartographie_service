import cv2
import numpy as np
import tempfile
import os
from .walls_doors_detection import remove_elements
from . import CLIENT, furniture_config

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