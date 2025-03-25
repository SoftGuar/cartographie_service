import cv2
import numpy as np

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