from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field, validator
import math
from database import get_db
from services.navigation_service import NavigationService

router = APIRouter()

class NavigationRequest(BaseModel):
    floor: str = Field(..., description="Name of the floor to navigate on")
    poi: str = Field(..., description="Name of the destination POI")
    poi1: Optional[str] = Field(None, description="Name of the starting POI (alternative to x_start/y_start)")
    x_start: Optional[float] = Field(None, description="Starting X coordinate (column) in the grid", ge=0)
    y_start: Optional[float] = Field(None, description="Starting Y coordinate (row) in the grid", ge=0)
    current_orientation: Optional[float] = Field(None, description="Current user orientation in degrees (0°=East, 90°=North, 180°=West, 270°=South)")

    @validator('x_start', 'y_start')
    def validate_coordinates(cls, v, values):
        if 'poi1' in values and values['poi1'] is not None:
            if v is not None:
                raise ValueError("Cannot specify both poi1 and x_start/y_start")
        return v

    @validator('poi1')
    def validate_poi1(cls, v, values):
        if v is None and (values.get('x_start') is None or values.get('y_start') is None):
            raise ValueError("Must specify either poi1 or both x_start and y_start")
        return v

    class Config:
        from_attributes = True
        json_schema_extra = {
            "example": {
                "floor": "Floor 1",
                "poi": "Reception",
                "poi1": "Entrance",
                "current_orientation": 30.0
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

    class Config:
        from_attributes = True
        json_schema_extra = {
            "example": {
                "actions": [
                    {
                        "type": "move",
                        "steps": 25,
                        "message": "Move 25 steps northwest"
                    },
                    {
                        "type": "rotate",
                        "degrees": 45,
                        "message": "Turn 45° right to face west"
                    }
                ],
                "total_actions": 2
            }
        }

class NavigationDebugResponse(BaseModel):
    """Extended response with debug information"""
    actions: List[ActionItem]
    total_actions: int
    estimated_distance: Optional[int]
    path_coordinates: List[Dict[str, int]]
    initial_orientation: float
    initial_direction_name: str
    debug_info: Dict[str, Any]

class ObstacleNavigationRequest(BaseModel):
    floor: str = Field(..., description="Name of the floor to navigate on")
    poi: str = Field(..., description="Name of the destination POI")
    x: float = Field(..., description="Current X coordinate in meters", ge=0)
    y: float = Field(..., description="Current Y coordinate in meters", ge=0)
    distance: float = Field(..., description="Distance in centimeters to avoid obstacles in front", ge=0)
    orientation: float = Field(..., description="Current user orientation in degrees (0°=East, 90°=North, 180°=West, 270°=South)")
    environment: str = Field(default="actions-test", description="Environment name for logging purposes")

    class Config:
        from_attributes = True
        json_schema_extra = {
            "example": {
                "floor": "Floor 1",
                "poi": "Reception",
                "x": 25.6,  # 25.6 meters
                "y": 9.6,   # 9.6 meters
                "distance": 100.0,  # 100 centimeters
                "orientation": 30.0,
                "environment": "actions-test"
            }
        }

@router.post(
    "/navigate", 
    response_model=NavigationResponse,
    summary="Get navigation instructions",
    description="Calculate navigation instructions from a starting position to a POI with obstacle avoidance"
)
def get_navigation_instructions(
    request: NavigationRequest,
    db: Session = Depends(get_db)
):
    """Get navigation instructions from a starting position to a POI"""
    try:
        navigation_service = NavigationService(db)
        
        # Get floor ID from floor name
        floor_id = navigation_service.get_floor_id_from_name(request.floor)
        if not floor_id:
            raise HTTPException(status_code=404, detail=f"Floor '{request.floor}' not found")
            
        # Get POI ID from POI name
        poi_id = navigation_service.get_poi_id_from_name(request.poi)
        if not poi_id:
            raise HTTPException(status_code=404, detail=f"Destination POI '{request.poi}' not found")
        
        # Determine start position
        if request.poi1:
            # Get start position from POI1
            start_poi_id = navigation_service.get_poi_id_from_name(request.poi1)
            if not start_poi_id:
                raise HTTPException(status_code=404, detail=f"Starting POI '{request.poi1}' not found")
            start_pos = navigation_service.get_poi_position(start_poi_id)
        else:
            # Use provided coordinates
            start_pos = (int(round(request.y_start)), int(round(request.x_start)))
        
        print(f"Converted coordinates: {start_pos}")
        
        actions, path, path_initial_orientation = navigation_service.get_navigation_actions_with_path(
            floor_id=floor_id,
            poi_id=poi_id,
            start_pos=start_pos
        )
        
        if not actions or not path:
            raise HTTPException(
                status_code=422, 
                detail="No safe navigation path found. This could be due to obstacles blocking the route or invalid start/destination positions."
            )
        
        # Handle initial orientation adjustment if provided
        final_actions = []
        if request.current_orientation is not None:
            # Calculate rotation needed to align user's current orientation with path's initial orientation
            initial_rotation = path_initial_orientation - request.current_orientation
            
            # Normalize to [-180, 180] for shortest rotation
            while initial_rotation > 180:
                initial_rotation -= 360
            while initial_rotation <= -180:
                initial_rotation += 360
            
            # Add initial alignment rotation if significant
            if abs(initial_rotation) > 5:  # Only if rotation is > 5 degrees
                direction_word = "left" if initial_rotation > 0 else "right"
                final_actions.append({
                    "type": "rotate",
                    "degrees": abs(int(round(initial_rotation))),
                    "message": f"Rotate {abs(int(round(initial_rotation)))}° {direction_word}"
                })
                print(f"Added initial alignment: {abs(initial_rotation):.0f}° {direction_word} (from {request.current_orientation}° to {path_initial_orientation}°)")
        
        # Add the calculated path actions
        final_actions.extend(actions)
        
        # Convert to response format
        action_items = [
            ActionItem(
                type=action["type"],
                steps=action.get("steps"),
                degrees=action.get("degrees"),
                message=action["message"]
            )
            for action in final_actions
        ]
        
        return NavigationResponse(
            actions=action_items,
            total_actions=len(action_items)
        )
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Navigation error: {str(e)}")

@router.post(
    "/navigate/detailed-path",
    summary="Get detailed maze navigation path",
    description="Returns every single grid cell the agent will pass through in the maze"
)
def get_detailed_navigation_path(
    request: NavigationRequest,
    db: Session = Depends(get_db)
):
    """Get the complete detailed path through the maze"""
    try:
        navigation_service = NavigationService(db)
        start_pos = (int(round(request.y_start)), int(round(request.x_start)))
        
        # Get floor grid and POI position
        grid, grid_dimensions = navigation_service.get_floor_grid(request.floor)
        goal_pos = navigation_service.get_poi_position(request.poi)
        
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
            
            # Track any obstacles the path goes through
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
        start_pos = (int(round(request.y_start)), int(round(request.x_start)))
        
        # Get grid and POI position
        grid, grid_dimensions = navigation_service.get_floor_grid(request.floor)
        goal_pos = navigation_service.get_poi_position(request.poi)
        
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
                "floor": request.floor,
                "poi": request.poi,
                "x_start": request.x_start,
                "y_start": request.y_start
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

@router.post(
    "/navigate/with-debug",
    response_model=NavigationDebugResponse,
    summary="Get navigation with detailed debug information",
    description="Returns navigation actions along with comprehensive debug information"
)
def get_navigation_with_debug(
    request: NavigationRequest,
    db: Session = Depends(get_db)
):
    """Get navigation instructions with full debug information"""
    try:
        navigation_service = NavigationService(db)
        start_pos = (int(round(request.y_start)), int(round(request.x_start)))
        
        # Get navigation with full debug output
        actions, path, initial_orientation = navigation_service.get_navigation_actions_with_path(
            floor_id=request.floor,
            poi_id=request.poi,
            start_pos=start_pos
        )
        
        if not actions or not path:
            raise HTTPException(
                status_code=422, 
                detail="No navigation path found"
            )
        
        # Get additional debug information
        grid, grid_dimensions = navigation_service.get_floor_grid(request.floor)
        goal_pos = navigation_service.get_poi_position(request.poi)
        
        # Calculate estimated total distance in steps
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
        
        # Convert path to coordinates
        path_coordinates = [{"x": point[1], "y": point[0]} for point in path]
        
        # Get direction name
        direction_name = navigation_service.get_direction_name(initial_orientation)
        
        # Comprehensive debug information
        debug_info = {
            "grid_dimensions": grid_dimensions,
            "start_position": {"row": start_pos[0], "col": start_pos[1]},
            "goal_position": {"row": goal_pos[0], "col": goal_pos[1]},
            "path_length": len(path),
            "path_validation": {
                "all_walkable": all(
                    grid[point[0]][point[1]] == 0 
                    for point in path 
                    if 0 <= point[0] < len(grid) and 0 <= point[1] < len(grid[0])
                ),
                "obstacles_in_path": [
                    {"step": i, "position": {"x": point[1], "y": point[0]}, "grid_value": grid[point[0]][point[1]]}
                    for i, point in enumerate(path)
                    if 0 <= point[0] < len(grid) and 0 <= point[1] < len(grid[0]) and grid[point[0]][point[1]] == 1
                ]
            },
            "margin_analysis": {
                "start_best_margin": navigation_service.find_best_valid_margin(grid, start_pos),
                "goal_best_margin": navigation_service.find_best_valid_margin(grid, goal_pos)
            },
            "distance_calculations": {
                "euclidean": math.sqrt((goal_pos[0] - start_pos[0])**2 + (goal_pos[1] - start_pos[1])**2),
                "manhattan": abs(goal_pos[0] - start_pos[0]) + abs(goal_pos[1] - start_pos[1])
            }
        }
        
        return NavigationDebugResponse(
            actions=action_items,
            total_actions=len(action_items),
            estimated_distance=estimated_distance,
            path_coordinates=path_coordinates,
            initial_orientation=initial_orientation,
            initial_direction_name=direction_name,
            debug_info=debug_info
        )
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Navigation with debug error: {str(e)}")

@router.post(
    "/test-pathfinding",
    summary="Test different pathfinding algorithms",
    description="Compare different pathfinding approaches for debugging"
)
def test_pathfinding_algorithms(
    request: NavigationRequest,
    db: Session = Depends(get_db)
):
    """Test and compare different pathfinding approaches"""
    try:
        navigation_service = NavigationService(db)
        start_pos = (int(round(request.y_start)), int(round(request.x_start)))
        
        # Get grid and goal
        grid, grid_dimensions = navigation_service.get_floor_grid(request.floor)
        goal_pos = navigation_service.get_poi_position(request.poi)
        
        results = {}
        
        # Test 1: Direct line path
        try:
            direct_path = navigation_service.bresenham_line(start_pos[0], start_pos[1], goal_pos[0], goal_pos[1])
            direct_valid = all(
                navigation_service.is_valid_position(grid, point, 0) 
                for point in direct_path
            )
            results["direct_line"] = {
                "path_length": len(direct_path),
                "is_valid": direct_valid,
                "path": [{"x": p[1], "y": p[0]} for p in direct_path[:10]]  # First 10 points only
            }
        except Exception as e:
            results["direct_line"] = {"error": str(e)}
        
        # Test 2: A* with different margins
        for margin in [0, 1, 2, 3]:
            try:
                path = navigation_service.find_path_raw(grid, start_pos, goal_pos, margin)
                if path:
                    results[f"astar_margin_{margin}"] = {
                        "path_length": len(path),
                        "success": True,
                        "path_sample": [{"x": p[1], "y": p[0]} for p in path[:5]]  # First 5 points
                    }
                else:
                    results[f"astar_margin_{margin}"] = {
                        "success": False,
                        "reason": "No path found"
                    }
            except Exception as e:
                results[f"astar_margin_{margin}"] = {"error": str(e)}
        
        # Test 3: Smoothed path
        try:
            smoothed_path = navigation_service.find_path(grid, start_pos, goal_pos)
            if smoothed_path:
                results["smoothed_path"] = {
                    "path_length": len(smoothed_path),
                    "success": True,
                    "full_path": [{"x": p[1], "y": p[0]} for p in smoothed_path]
                }
            else:
                results["smoothed_path"] = {
                    "success": False,
                    "reason": "No smoothed path found"
                }
        except Exception as e:
            results["smoothed_path"] = {"error": str(e)}
        
        # Test 4: Complete detailed path
        try:
            detailed_path = navigation_service.get_complete_path(grid, start_pos, goal_pos)
            if detailed_path:
                results["detailed_path"] = {
                    "path_length": len(detailed_path),
                    "success": True,
                    "path_sample": [{"x": p[1], "y": p[0]} for p in detailed_path[:10]]  # First 10 points
                }
            else:
                results["detailed_path"] = {
                    "success": False,
                    "reason": "No detailed path found"
                }
        except Exception as e:
            results["detailed_path"] = {"error": str(e)}
        
        return {
            "test_info": {
                "start_position": {"row": start_pos[0], "col": start_pos[1]},
                "goal_position": {"row": goal_pos[0], "col": goal_pos[1]},
                "grid_dimensions": grid_dimensions,
                "euclidean_distance": math.sqrt((goal_pos[0] - start_pos[0])**2 + (goal_pos[1] - start_pos[1])**2)
            },
            "pathfinding_results": results
        }
        
    except Exception as e:
        return {"error": str(e)}

@router.post(
    "/navigate/obstacle", 
    response_model=NavigationResponse,
    summary="Get navigation instructions with obstacle avoidance",
    description="Calculate navigation instructions avoiding obstacles in front of the user. Coordinates are in meters, distance in centimeters."
)
def get_obstacle_navigation_instructions(
    request: ObstacleNavigationRequest,
    db: Session = Depends(get_db)
):
    """Get navigation instructions avoiding obstacles in front of the user"""
    try:
        navigation_service = NavigationService(db)
        
        # Get floor ID from floor name
        floor_id = navigation_service.get_floor_id_from_name(request.floor)
        if not floor_id:
            raise HTTPException(status_code=404, detail=f"Floor '{request.floor}' not found")
            
        # Get POI ID from POI name
        poi_id = navigation_service.get_poi_id_from_name(request.poi)
        if not poi_id:
            raise HTTPException(status_code=404, detail=f"Destination POI '{request.poi}' not found")
        
        # Convert meters to grid cells (1 grid cell = 20cm = 0.2 meters)
        GRID_CELL_SIZE_METERS = 0.2
        start_pos = (
            int(round(request.y / GRID_CELL_SIZE_METERS)),  # Convert meters to grid rows
            int(round(request.x / GRID_CELL_SIZE_METERS))   # Convert meters to grid columns
        )
        
        # Get floor grid
        grid, grid_dimensions = navigation_service.get_floor_grid(floor_id)
        
        # Convert distance from centimeters to grid cells (1 grid cell = 20cm)
        GRID_CELL_SIZE_CM = 20
        grid_cells_distance = int(round(request.distance / GRID_CELL_SIZE_CM))
        
        # Calculate the grid positions to avoid based on orientation
        # Convert orientation to radians
        orientation_rad = math.radians(request.orientation)
        
        # Calculate the direction vector
        dx = math.cos(orientation_rad)
        dy = -math.sin(orientation_rad)  # Negative because y increases downward
        
        # Calculate the positions to avoid (4 squares in front)
        positions_to_avoid = []
        for i in range(4):  # 4 squares to avoid
            # Calculate position at the specified distance
            avoid_x = int(round(start_pos[1] + dx * (grid_cells_distance + i)))
            avoid_y = int(round(start_pos[0] + dy * (grid_cells_distance + i)))
            
            # Add the position if it's within grid bounds
            if 0 <= avoid_x < grid_dimensions[1] and 0 <= avoid_y < grid_dimensions[0]:
                positions_to_avoid.append((avoid_y, avoid_x))
        
        # Mark these positions as obstacles in the grid
        for pos in positions_to_avoid:
            if 0 <= pos[0] < len(grid) and 0 <= pos[1] < len(grid[0]):
                grid[pos[0]][pos[1]] = 1
        
        # Get navigation actions with the modified grid
        actions, path, path_initial_orientation = navigation_service.get_navigation_actions_with_path(
            floor_id=floor_id,
            poi_id=poi_id,
            start_pos=start_pos,
            custom_grid=grid  # Pass the modified grid
        )
        
        if not actions or not path:
            raise HTTPException(
                status_code=422, 
                detail="No safe navigation path found. This could be due to obstacles blocking the route or invalid start/destination positions."
            )
        
        # Handle initial orientation adjustment if provided
        final_actions = []
        if request.orientation is not None:
            # Calculate rotation needed to align user's current orientation with path's initial orientation
            initial_rotation = path_initial_orientation - request.orientation
            
            # Normalize to [-180, 180] for shortest rotation
            while initial_rotation > 180:
                initial_rotation -= 360
            while initial_rotation <= -180:
                initial_rotation += 360
            
            # Add initial alignment rotation if significant
            if abs(initial_rotation) > 5:  # Only if rotation is > 5 degrees
                direction_word = "left" if initial_rotation > 0 else "right"
                final_actions.append({
                    "type": "rotate",
                    "degrees": abs(int(round(initial_rotation))),
                    "message": f"Rotate {abs(int(round(initial_rotation)))}° {direction_word}"
                })
        
        # Add the calculated path actions
        final_actions.extend(actions)
        
        # Convert to response format
        action_items = [
            ActionItem(
                type=action["type"],
                steps=action.get("steps"),
                degrees=action.get("degrees"),
                message=action["message"]
            )
            for action in final_actions
        ]
        
        return NavigationResponse(
            actions=action_items,
            total_actions=len(action_items)
        )
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Navigation error: {str(e)}")