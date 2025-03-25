import base64
import binascii

def validate_base64(image_data: str) -> bool:
    """Validate base64 image data"""
    if not image_data:
        return True
    if ',' not in image_data:
        return False
    try:
        base64.b64decode(image_data.split(',')[1])
        return True
    except (binascii.Error, ValueError):
        return False
