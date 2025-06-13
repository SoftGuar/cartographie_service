from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import math
from database import get_db
from services.navigation_service import NavigationService

router = APIRouter()

class NavigationRequest(BaseModel):
    floor_id: str = Field(..., description="ID of the floor to navigate on")
    poi_id: str = Field(..., description="ID of the destination POI")
    start_x: float = Field(..., description="Starting X coordinate (column) in the grid", ge=0)
    start_y: float = Field(..., description="Starting Y coordinate (row) in the grid", ge=0)

    class Config:
        json_schema_extra = {
            "example": {
                "floor_id": "floor-123",
                "poi_id": "poi-456",
                "start_x": 128.0,
                "start_y": 48.0
            }
        }

class ActionItem(BaseModel):
    type: str = Field(..., description="Type of action: 'move' or 'rotate'")
    steps: Optional[int] = Field(None, description="Number of steps to move (for move actions)")
    degrees: Optional[int] = Field(None, description="Degrees to rotate (for rotate actions)")
    message: str = Field(..., description="Human-readable description of the action")

class NavigationResponse(BaseModel):
    actions: List[ActionItem] = Field(..., description="List of navigation actions")
    total_actions: int = Field(..., description="Total number of actions")
    estimated_distance: Optional[int] = Field(None, description="Estimated total distance in grid units")
    path_coordinates: List[Dict[str, int]] = Field(default=[], description="Complete path as x,y coordinates")
    
    class Config:
        json_schema_extra = {
            "example": {
                "actions": [
                    {
                        "type": "move",
                        "steps": 5,
                        "message": "Move forward 5 steps"
                    },
                    {
                        "type": "rotate",
                        "degrees": 90,
                        "message": "Rotate 90 degrees right"
                    },
                    {
                        "type": "move",
                        "steps": 10,
                        "message": "Move forward 10 steps"
                    }
                ],
                "total_actions": 3,
                "estimated_distance": 15,
                "path_coordinates": [
                    {"x": 10, "y": 5},
                    {"x": 11, "y": 5},
                    {"x": 12, "y": 5}
                ]
            }
        }

class NavigationDebugResponse(BaseModel):
    """Extended response with debug information"""
    actions: List[ActionItem]
    total_actions: int
    estimated_distance: Optional[int]
    path_coordinates: List[Dict[str, int]]
    debug_info: Dict[str, Any]

@router.post(
    "/navigate/detailed-path",
    summary="Get detailed maze navigation path",
    description="Returns every single grid cell the agent will pass through in the maze"
)
def get_detailed_navigation_path(
    request: NavigationRequest,
    db: Session = Depends(get_db)
):
    """
    Get the complete detailed path through the maze.
    Returns every single grid cell the agent will pass through.
    
    This is useful for visualizing the exact route and ensuring
    the agent doesn't pass through any obstacles (red areas).
    """
    try:
        navigation_service = NavigationService(db)
        start_pos = (int(round(request.start_y)), int(round(request.start_x)))
        
        # Get floor grid and POI position
        grid, grid_dimensions = navigation_service.get_floor_grid(request.floor_id)
        goal_pos = navigation_service.get_poi_position(request.poi_id)
        
        print(f"Getting detailed path from {start_pos} to {goal_pos}")
        
        # Get the complete detailed path
        detailed_path = navigation_service.get_complete_path(grid, start_pos, goal_pos)
        
        if not detailed_path:
            raise HTTPException(
                status_code=422, 
                detail="No valid path found through the maze. The route may be blocked by obstacles."
            )
        
        # Convert to coordinates and add grid values for validation
        path_with_validation = []
        obstacles_encountered = []
        
        for i, (row, col) in enumerate(detailed_path):
            grid_value = grid[row][col] if 0 <= row < len(grid) and 0 <= col < len(grid[0]) else -1
            
            path_step = {
                "step": i,
                "x": col,
                "y": row,
                "grid_value": grid_value,
                "is_walkable": grid_value == 0,
                "is_start": i == 0,
                "is_goal": i == len(detailed_path) - 1
            }
            
            path_with_validation.append(path_step)
            
            # Track any obstacles the path goes through (this should not happen!)
            if grid_value == 1:
                obstacles_encountered.append({
                    "step": i,
                    "position": {"x": col, "y": row},
                    "grid_value": grid_value
                })
        
        # Calculate simple movement instructions
        movements = []
        for i in range(1, len(detailed_path)):
            prev_row, prev_col = detailed_path[i-1]
            curr_row, curr_col = detailed_path[i]
            
            row_diff = curr_row - prev_row
            col_diff = curr_col - prev_col
            
            # Determine direction
            if row_diff == -1 and col_diff == 0:
                direction = "North (Up)"
            elif row_diff == 1 and col_diff == 0:
                direction = "South (Down)"
            elif row_diff == 0 and col_diff == 1:
                direction = "East (Right)"
            elif row_diff == 0 and col_diff == -1:
                direction = "West (Left)"
            elif row_diff == -1 and col_diff == 1:
                direction = "Northeast (Up-Right)"
            elif row_diff == -1 and col_diff == -1:
                direction = "Northwest (Up-Left)"
            elif row_diff == 1 and col_diff == 1:
                direction = "Southeast (Down-Right)"
            elif row_diff == 1 and col_diff == -1:
                direction = "Southwest (Down-Left)"
            else:
                direction = f"Unknown ({row_diff}, {col_diff})"
            
            movements.append({
                "step": i,
                "from": {"x": prev_col, "y": prev_row},
                "to": {"x": curr_col, "y": curr_row},
                "direction": direction,
                "movement": f"From ({prev_col},{prev_row}) to ({curr_col},{curr_row}) - {direction}"
            })
        
        return {
            "success": True,
            "total_steps": len(detailed_path),
            "start_position": {"x": detailed_path[0][1], "y": detailed_path[0][0]},
            "goal_position": {"x": detailed_path[-1][1], "y": detailed_path[-1][0]},
            "validation": {
                "path_is_valid": len(obstacles_encountered) == 0,
                "obstacles_encountered": obstacles_encountered,
                "total_obstacles": len(obstacles_encountered)
            },
            "complete_path": path_with_validation,
            "movements": movements,
            "grid_info": {
                "dimensions": {"rows": grid_dimensions[0], "cols": grid_dimensions[1]},
                "obstacle_count": sum(sum(1 for cell in row if cell == 1) for row in grid),
                "walkable_count": sum(sum(1 for cell in row if cell == 0) for row in grid)
            }
        }
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detailed navigation error: {str(e)}")

@router.post(
    "/navigate", 
    response_model=NavigationResponse,
    summary="Get navigation instructions",
    description="Calculate navigation instructions from a starting position to a POI with obstacle avoidance",
    responses={
        400: {"description": "Invalid request parameters"},
        404: {"description": "Floor or POI not found"},
        422: {"description": "No valid path found"},
        500: {"description": "Internal server error"}
    }
)
def get_navigation_instructions(
    request: NavigationRequest,
    db: Session = Depends(get_db)
):
    """
    Get navigation instructions from a starting position to a POI.
    
    The navigation system:
    - Uses A* pathfinding algorithm for optimal routes
    - Maintains a safety margin from obstacles (configurable, default 3 grid cells)
    - Supports both straight and diagonal movement for smoother paths
    - 1 step = 2 grid squares for realistic movement
    - Only rotates for direction changes > 30 degrees
    - Returns empty action list if no safe path is found
    
    Args:
        request: NavigationRequest containing floor_id, poi_id, start_x, start_y
        
    Returns:
        NavigationResponse with list of actions and metadata
    """
    try:
        navigation_service = NavigationService(db)
        
        # Convert float coordinates to integers (note: start_x is column, start_y is row)
        start_pos = (int(round(request.start_y)), int(round(request.start_x)))
        
        print(f"Converted coordinates: ({request.start_x}, {request.start_y}) -> grid position {start_pos}")
        
        actions, path = navigation_service.get_navigation_actions_with_path(
            floor_id=request.floor_id,
            poi_id=request.poi_id,
            start_pos=start_pos
        )
        
        if not actions or not path:
            raise HTTPException(
                status_code=422, 
                detail="No safe navigation path found. This could be due to obstacles blocking the route or invalid start/destination positions."
            )
        
        # Calculate estimated total distance in steps (remember: 1 step = 2 grid squares)
        estimated_distance = sum(
            action.get("steps", 0) for action in actions if action.get("type") == "move"
        )
        
        # Convert to response format
        action_items = [
            ActionItem(
                type=action["type"],
                steps=action.get("steps"),
                degrees=action.get("degrees"),
                message=action["message"]
            )
            for action in actions
        ]
        
        # Convert path to coordinates (row, col) -> (x, y)
        path_coordinates = [{"x": point[1], "y": point[0]} for point in path]
        
        return NavigationResponse(
            actions=action_items,
            total_actions=len(action_items),
            estimated_distance=estimated_distance,
            path_coordinates=path_coordinates
        )
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Navigation error: {str(e)}")

@router.post(
    "/navigate/debug", 
    response_model=NavigationDebugResponse,
    summary="Get navigation instructions with debug info",
    description="Same as /navigate but includes additional debug information"
)
def get_navigation_instructions_debug(
    request: NavigationRequest,
    db: Session = Depends(get_db)
):
    """Get navigation instructions with additional debug information"""
    try:
        navigation_service = NavigationService(db)
        start_pos = (int(round(request.start_y)), int(round(request.start_x)))
        
        actions, path = navigation_service.get_navigation_actions_with_path(
            floor_id=request.floor_id,
            poi_id=request.poi_id,
            start_pos=start_pos
        )
        
        estimated_distance = sum(
            action.get("steps", 0) for action in actions if action.get("type") == "move"
        )
        
        action_items = [
            ActionItem(
                type=action["type"],
                steps=action.get("steps"),
                degrees=action.get("degrees"),
                message=action["message"]
            )
            for action in actions
        ]
        
        # Get grid info and positions for debug
        grid, grid_dimensions = navigation_service.get_floor_grid(request.floor_id)
        goal_pos = navigation_service.get_poi_position(request.poi_id)
        
        debug_info = {
            "start_position": {"row": start_pos[0], "col": start_pos[1]},
            "goal_position": {"row": goal_pos[0], "col": goal_pos[1]},
            "grid_dimensions": {"rows": grid_dimensions[0], "cols": grid_dimensions[1]},
            "obstacle_margin": 3,  # From NavigationService.OBSTACLE_MARGIN
            "start_valid": navigation_service.is_valid_position(grid, start_pos),
            "goal_valid": navigation_service.is_valid_position(grid, goal_pos),
            "path_found": len(actions) > 0,
            "path_length": len(path) if path else 0
        }
        
        # Convert path to coordinates (row, col) -> (x, y)
        path_coordinates = [{"x": point[1], "y": point[0]} for point in path] if path else []
        
        return NavigationDebugResponse(
            actions=action_items,
            total_actions=len(action_items),
            estimated_distance=estimated_distance,
            path_coordinates=path_coordinates,
            debug_info=debug_info
        )
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Navigation error: {str(e)}")

@router.post(
    "/debug-positions",
    summary="Debug start and destination positions",
    description="Check the validity of start and destination positions without pathfinding"
)
def debug_positions(
    request: NavigationRequest,
    db: Session = Depends(get_db)
):
    """Debug endpoint to check position validity and grid data"""
    try:
        navigation_service = NavigationService(db)
        start_pos = (int(round(request.start_y)), int(round(request.start_x)))
        
        # Get grid and POI position
        grid, grid_dimensions = navigation_service.get_floor_grid(request.floor_id)
        goal_pos = navigation_service.get_poi_position(request.poi_id)
        
        rows, cols = len(grid), len(grid[0]) if grid else 0
        
        # Check positions with different margins
        start_validity = {}
        goal_validity = {}
        
        for margin in [0, 1, 2, 3, 4, 5]:
            start_validity[f"margin_{margin}"] = navigation_service.is_valid_position(grid, start_pos, margin)
            goal_validity[f"margin_{margin}"] = navigation_service.is_valid_position(grid, goal_pos, margin)
        
        # Get surrounding areas
        def get_surrounding_area(pos, size=5):
            area = []
            for dr in range(-size//2, size//2 + 1):
                row_data = []
                for dc in range(-size//2, size//2 + 1):
                    check_row, check_col = pos[0] + dr, pos[1] + dc
                    if 0 <= check_row < rows and 0 <= check_col < cols:
                        row_data.append(grid[check_row][check_col])
                    else:
                        row_data.append(-1)  # Out of bounds
                area.append(row_data)
            return area
        
        # Calculate distance
        row_diff = goal_pos[0] - start_pos[0]
        col_diff = goal_pos[1] - start_pos[1]
        euclidean_distance = math.sqrt(row_diff**2 + col_diff**2)
        
        return {
            "request": {
                "floor_id": request.floor_id,
                "poi_id": request.poi_id,
                "start_x": request.start_x,
                "start_y": request.start_y
            },
            "converted_positions": {
                "start_grid_pos": {"row": start_pos[0], "col": start_pos[1]},
                "goal_grid_pos": {"row": goal_pos[0], "col": goal_pos[1]}
            },
            "grid_info": {
                "dimensions": {"rows": rows, "cols": cols},
                "grid_dimensions_from_db": grid_dimensions
            },
            "distance": {
                "row_difference": row_diff,
                "col_difference": col_diff,
                "euclidean_distance": round(euclidean_distance, 2),
                "expected_steps_min": round(euclidean_distance / 2, 1)
            },
            "position_validity": {
                "start_position": start_validity,
                "goal_position": goal_validity
            },
            "grid_values": {
                "at_start": grid[start_pos[0]][start_pos[1]] if 0 <= start_pos[0] < rows and 0 <= start_pos[1] < cols else "out_of_bounds",
                "at_goal": grid[goal_pos[0]][goal_pos[1]] if 0 <= goal_pos[0] < rows and 0 <= goal_pos[1] < cols else "out_of_bounds"
            },
            "surrounding_areas": {
                "start_5x5": get_surrounding_area(start_pos, 5),
                "goal_5x5": get_surrounding_area(goal_pos, 5)
            },
            "grid_sample_10x10": [
                grid[i][:10] if i < len(grid) and len(grid[i]) >= 10 else (grid[i] if i < len(grid) else [])
                for i in range(min(10, len(grid)))
            ]
        }
        
    except Exception as e:
        return {"error": str(e)}