from typing import List, Dict, Any, Tuple, Optional
import json
import numpy as np
import math
import heapq
from sqlalchemy.orm import Session
from fastapi import HTTPException

from models.floor import Floor
from models.poi import POI
from models.point import Point

# Constants
GRID_CELL_SIZE = 0.7  # Each grid cell is 0.7 meters

class NavigationService:
    """Service class for handling indoor navigation functionality."""
    
    @staticmethod
    def real_to_grid(x: float, y: float) -> Tuple[int, int]:
        """Convert real-world coordinates (meters) to grid coordinates."""
        grid_x = int(x / GRID_CELL_SIZE)
        grid_y = int(y / GRID_CELL_SIZE)
        return (grid_x, grid_y)

    @staticmethod
    def grid_to_real(grid_x: int, grid_y: int) -> Tuple[float, float]:
        """Convert grid coordinates to real-world coordinates (meters)."""
        real_x = (grid_x + 0.5) * GRID_CELL_SIZE  # Center of the grid cell
        real_y = (grid_y + 0.5) * GRID_CELL_SIZE  # Center of the grid cell
        return (real_x, real_y)

    @staticmethod
    def parse_grid_data(grid_data_str: str) -> np.ndarray:
        """Parse grid data string into numpy array."""
        try:
            grid_data = json.loads(grid_data_str)
            return np.array(grid_data)
        except (json.JSONDecodeError, ValueError):
            raise ValueError("Invalid grid data format")

    @staticmethod
    def calculate_angle(start_x: float, start_y: float, end_x: float, end_y: float) -> float:
        """Calculate angle in degrees from start point to end point."""
        dx = end_x - start_x
        dy = end_y - start_y
        angle = math.degrees(math.atan2(dy, dx))
        # Normalize to 0-360 range
        return (angle + 360) % 360

    @staticmethod
    def calculate_direction(angle: float) -> str:
        """Convert angle to human-readable direction."""
        directions = ["east", "northeast", "north", "northwest", 
                    "west", "southwest", "south", "southeast"]
        index = round(angle / 45) % 8
        return directions[index]

    @staticmethod
    def heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """Calculate Manhattan distance heuristic for A* algorithm."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    @classmethod
    def a_star_search(cls, grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        A* pathfinding algorithm to find shortest path from start to goal.
        
        Args:
            grid: 2D numpy array where 0 = walkable, 1 = obstacle
            start: Starting grid position (x, y)
            goal: Goal grid position (x, y)
            
        Returns:
            List of grid coordinates representing the path
        """
        if not (0 <= start[0] < grid.shape[1] and 0 <= start[1] < grid.shape[0]):
            raise ValueError(f"Start position {start} is outside grid boundaries")
        
        if not (0 <= goal[0] < grid.shape[1] and 0 <= goal[1] < grid.shape[0]):
            raise ValueError(f"Goal position {goal} is outside grid boundaries")
        
        if grid[start[1], start[0]] == 1:
            raise ValueError(f"Start position {start} is on an obstacle")
            
        if grid[goal[1], goal[0]] == 1:
            raise ValueError(f"Goal position {goal} is on an obstacle")
        
        # Neighbors in 4 directions (up, right, down, left)
        neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        open_set = []
        closed_set = set()
        
        # Priority queue with (f_score, position)
        heapq.heappush(open_set, (0, start))
        
        came_from = {}
        g_score = {start: 0}
        f_score = {start: cls.heuristic(start, goal)}
        
        while open_set:
            _, current = heapq.heappop(open_set)
            
            if current == goal:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()
                return path
            
            closed_set.add(current)
            
            for dx, dy in neighbors:
                neighbor = (current[0] + dx, current[1] + dy)
                
                # Check if neighbor is valid (within grid and not an obstacle)
                if not (0 <= neighbor[0] < grid.shape[1] and 0 <= neighbor[1] < grid.shape[0]):
                    continue
                    
                if grid[neighbor[1], neighbor[0]] == 1:
                    continue
                    
                if neighbor in closed_set:
                    continue
                
                tentative_g_score = g_score[current] + 1
                
                if neighbor not in [item[1] for item in open_set] or tentative_g_score < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f = tentative_g_score + cls.heuristic(neighbor, goal)
                    f_score[neighbor] = f
                    
                    if neighbor not in [item[1] for item in open_set]:
                        heapq.heappush(open_set, (f, neighbor))
        
        # No path found
        return []

    @classmethod
    def generate_actions(cls, path: List[Tuple[int, int]]) -> List[Dict[str, Any]]:
        """
        Generate navigation actions from a grid path.
        
        Args:
            path: List of grid positions [(x, y), ...]
            
        Returns:
            List of action dictionaries
        """
        if not path or len(path) < 2:
            return []
        
        actions = []
        current_direction = None
        steps_count = 0
        
        for i in range(1, len(path)):
            prev_grid_x, prev_grid_y = path[i-1]
            curr_grid_x, curr_grid_y = path[i]
            
            # Calculate direction vector
            dx = curr_grid_x - prev_grid_x
            dy = curr_grid_y - prev_grid_y
            
            # Determine new direction
            new_direction = None
            if dx == 1 and dy == 0:
                new_direction = "east"
            elif dx == -1 and dy == 0:
                new_direction = "west"
            elif dx == 0 and dy == 1:
                new_direction = "south"
            elif dx == 0 and dy == -1:
                new_direction = "north"
            
            # If direction changed, add rotation action
            if current_direction is not None and new_direction != current_direction:
                # Add move action for accumulated steps
                if steps_count > 0:
                    prev_real_x, prev_real_y = cls.grid_to_real(path[i-steps_count-1][0], path[i-steps_count-1][1])
                    curr_real_x, curr_real_y = cls.grid_to_real(prev_grid_x, prev_grid_y)
                    
                    actions.append({
                        "type": "move",
                        "value": steps_count * GRID_CELL_SIZE,
                        "message": f"Walk {steps_count * GRID_CELL_SIZE:.1f} meters {current_direction}",
                        "end_position": {"x": curr_real_x, "y": curr_real_y}
                    })
                    steps_count = 0
                
                # Calculate rotation angle
                prev_real_x, prev_real_y = cls.grid_to_real(prev_grid_x, prev_grid_y)
                curr_real_x, curr_real_y = cls.grid_to_real(curr_grid_x, curr_grid_y)
                angle = cls.calculate_angle(prev_real_x, prev_real_y, curr_real_x, curr_real_y)
                
                actions.append({
                    "type": "rotate",
                    "value": angle,
                    "message": f"Turn to face {new_direction}",
                    "end_position": {"x": prev_real_x, "y": prev_real_y}
                })
            
            # Increment steps in current direction
            steps_count += 1
            current_direction = new_direction
            
            # If this is the last point, add the final move action
            if i == len(path) - 1 and steps_count > 0:
                prev_real_x, prev_real_y = cls.grid_to_real(path[i-steps_count][0], path[i-steps_count][1])
                curr_real_x, curr_real_y = cls.grid_to_real(curr_grid_x, curr_grid_y)
                
                actions.append({
                    "type": "move",
                    "value": steps_count * GRID_CELL_SIZE,
                    "message": f"Walk {steps_count * GRID_CELL_SIZE:.1f} meters {current_direction}",
                    "end_position": {"x": curr_real_x, "y": curr_real_y}
                })
        
        return actions

    @classmethod
    def get_navigation_path(cls, db: Session, current_x: float, current_y: float, 
                        floor_id: str, destination_poi_id: str) -> Dict[str, Any]:
        """
        Calculate navigation path from current position to destination POI.
        
        Args:
            db: Database session
            current_x: Current real-world x coordinate in meters
            current_y: Current real-world y coordinate in meters
            floor_id: ID of the floor
            destination_poi_id: ID of the destination POI
            
        Returns:
            Dictionary with path, actions, and total distance
        """
        # Get floor data
        floor = db.query(Floor).filter(Floor.id == floor_id).first()
        if not floor:
            raise HTTPException(status_code=404, detail=f"Floor with ID {floor_id} not found")
        
        # Get destination POI data
        poi = db.query(POI).filter(POI.id == destination_poi_id).first()
        if not poi:
            raise HTTPException(status_code=404, detail=f"POI with ID {destination_poi_id} not found")
        
        # Get POI's point data
        point = db.query(Point).filter(Point.id == poi.point_id).first()
        if not point:
            raise HTTPException(status_code=404, detail=f"Point for POI not found")
        
        # Parse floor grid data
        grid = cls.parse_grid_data(floor.grid_data)
        
        # Convert real-world coordinates to grid coordinates
        start_grid_x, start_grid_y = cls.real_to_grid(current_x, current_y)
        goal_grid_x, goal_grid_y = cls.real_to_grid(point.x, point.y)
        
        # Find path using A* algorithm
        grid_path = cls.a_star_search(grid, (start_grid_x, start_grid_y), (goal_grid_x, goal_grid_y))
        
        if not grid_path:
            raise HTTPException(status_code=404, detail="No path found from current position to destination")
        
        # Generate navigation actions
        actions = cls.generate_actions(grid_path)
        
        # Calculate total distance
        total_distance = 0
        for action in actions:
            if action["type"] == "move":
                total_distance += action["value"]
        
        # Prepare response
        path_coords = [{"x": x, "y": y} for x, y in grid_path]
        
        return {
            "path": path_coords,
            "actions": actions,
            "total_distance": total_distance
        }