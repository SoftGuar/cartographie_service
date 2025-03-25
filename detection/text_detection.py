import cv2
import numpy as np
import tempfile
import os
from . import CLIENT, text_config

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