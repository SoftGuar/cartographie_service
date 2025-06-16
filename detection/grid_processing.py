import cv2
import numpy as np

def combine_final_result(original_image, no_text_image, walls_doors_mask, furniture_mask, grid_size=1):
    """
    Creates a final binary image with walls, doors, and furniture as black (1)
    and empty spaces as white (0), divided into a fixed grid of 62x40 cells.

    Args:
        original_image: Original floor plan image
        no_text_image: Image with text removed
        walls_doors_mask: Binary mask for walls and doors
        furniture_mask: Binary mask for furniture
        grid_size: Not used anymore, kept for backward compatibility
    """
    # Fixed grid dimensions
    GRID_COLS = 32
    GRID_ROWS = 20

    # Resize all input images to match the grid dimensions
    height, width = original_image.shape[:2]
    aspect_ratio = width / height
    target_width = GRID_COLS
    target_height = GRID_ROWS

    # Resize masks to match grid dimensions
    walls_doors_mask = cv2.resize(walls_doors_mask, (target_width, target_height), interpolation=cv2.INTER_NEAREST)
    furniture_mask = cv2.resize(furniture_mask, (target_width, target_height), interpolation=cv2.INTER_NEAREST)

    # Create a combined binary mask for walls, doors, and furniture
    combined_mask = cv2.bitwise_or(walls_doors_mask, furniture_mask)

    # Create an empty grid (0 = free space, 1 = obstacle)
    grid = np.zeros((GRID_ROWS, GRID_COLS), dtype=np.uint8)

    # For each grid cell, check if it contains any obstacle
    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            # If any pixel in this cell is an obstacle (255), mark the cell as obstacle (1)
            if combined_mask[row, col] > 0:
                grid[row, col] = 1

    # Create a visual representation of the grid
    grid_visual = np.ones((GRID_ROWS, GRID_COLS), dtype=np.uint8) * 255

    # Fill grid cells with black if they contain obstacles
    grid_visual[grid == 1] = 0

    # Create grid with lines for visualization
    grid_with_lines = grid_visual.copy()

    # Draw horizontal grid lines
    for row in range(GRID_ROWS + 1):
        cv2.line(grid_with_lines, (0, row), (GRID_COLS, row), 128, 1)

    # Draw vertical grid lines
    for col in range(GRID_COLS + 1):
        cv2.line(grid_with_lines, (col, 0), (col, GRID_ROWS), 128, 1)

    return grid_visual, grid, grid_with_lines