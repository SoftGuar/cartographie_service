import json
import math
from typing import List, Tuple, Optional
from heapq import heappush, heappop
from models.floor import Floor
from models.poi import POI
from models.point import Point
from sqlalchemy.orm import Session

# Global configuration
OBSTACLE_MARGIN = 3  # Minimum distance from obstacles in grid cells
DIAGONAL_COST = 1.414  # Cost for diagonal movement (sqrt(2))
STRAIGHT_COST = 1.0    # Cost for straight movement

class NavigationService:
    def __init__(self, db: Session):
        self.db = db

    def get_floor_grid(self, floor_id: str) -> Tuple[List[List[int]], Tuple[int, int]]:
        """Get the grid data and dimensions for a floor"""
        floor = self.db.query(Floor).filter(Floor.id == floor_id).first()
        if not floor:
            raise ValueError(f"Floor with id {floor_id} not found")
        
        grid_data = json.loads(floor.grid_data)
        grid_dimensions_raw = json.loads(floor.grid_dimensions)
        
        # Database stores as [cols, rows] but we need (rows, cols)
        actual_rows = len(grid_data)
        actual_cols = len(grid_data[0]) if grid_data else 0
        db_cols, db_rows = grid_dimensions_raw[0], grid_dimensions_raw[1]
        
        print(f"Grid dimensions analysis:")
        print(f"  Database stored: [cols={db_cols}, rows={db_rows}]")
        print(f"  Actual grid size: {actual_rows} rows x {actual_cols} cols")
        print(f"  Returning: (rows={actual_rows}, cols={actual_cols})")
        
        return grid_data, (actual_rows, actual_cols)

    def get_poi_position(self, poi_id: str) -> Tuple[int, int]:
        """Get the grid position (row, col) of a POI"""
        poi = self.db.query(POI).filter(POI.id == poi_id).first()
        if not poi:
            raise ValueError(f"POI with id {poi_id} not found")
        
        point = self.db.query(Point).filter(Point.id == poi.point_id).first()
        if not point:
            raise ValueError(f"Point with id {poi.point_id} not found")
        
        # Convert coordinates to integers and return as (row, col)
        row = int(round(point.y))
        col = int(round(point.x))
        print(f"POI position: point.x={point.x}, point.y={point.y} -> grid(row={row}, col={col})")
        return row, col

    def is_valid_position(self, grid: List[List[int]], pos: Tuple[int, int], margin: int = OBSTACLE_MARGIN) -> bool:
        """Check if a position is valid considering the obstacle margin"""
        rows, cols = len(grid), len(grid[0]) if grid else 0
        row, col = pos
        
        # Check if position is within grid bounds
        if not (0 <= row < rows and 0 <= col < cols):
            return False
            
        # Check if position itself is an obstacle (1 = obstacle, 0 = walkable)
        if grid[row][col] == 1:
            return False
            
        # Check margin around the position for obstacles
        for dr in range(-margin, margin + 1):
            for dc in range(-margin, margin + 1):
                check_row, check_col = row + dr, col + dc
                if (0 <= check_row < rows and 0 <= check_col < cols and 
                    grid[check_row][check_col] == 1):  # 1 = obstacle
                    return False
        
        return True

    def get_neighbors_with_margin(self, grid: List[List[int]], pos: Tuple[int, int], margin: int) -> List[Tuple[Tuple[int, int], float]]:
        """Get valid neighboring positions with specified margin"""
        row, col = pos
        neighbors = []
        
        # 8-directional movement (including diagonals for smoother paths)
        directions = [
            (0, 1, STRAIGHT_COST),   # Right
            (1, 0, STRAIGHT_COST),   # Down
            (0, -1, STRAIGHT_COST),  # Left
            (-1, 0, STRAIGHT_COST),  # Up
            (1, 1, DIAGONAL_COST),   # Down-Right
            (1, -1, DIAGONAL_COST),  # Down-Left
            (-1, 1, DIAGONAL_COST),  # Up-Right
            (-1, -1, DIAGONAL_COST)  # Up-Left
        ]
        
        for dr, dc, cost in directions:
            new_pos = (row + dr, col + dc)
            if self.is_valid_position(grid, new_pos, margin):
                neighbors.append((new_pos, cost))
        
        # Debug for positions with no neighbors
        if len(neighbors) == 0:
            print(f"WARNING: Position {pos} has NO valid neighbors with margin {margin}")
                
        return neighbors

    def get_neighbors(self, grid: List[List[int]], pos: Tuple[int, int]) -> List[Tuple[Tuple[int, int], float]]:
        """Get valid neighboring positions with default margin"""
        return self.get_neighbors_with_margin(grid, pos, OBSTACLE_MARGIN)

    def heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """Calculate Euclidean distance between two points"""
        return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

    def find_best_valid_margin(self, grid: List[List[int]], pos: Tuple[int, int], max_margin: int = 5) -> int:
        """Find the maximum valid margin for a position"""
        for margin in range(max_margin, -1, -1):
            if self.is_valid_position(grid, pos, margin):
                return margin
        return -1

    def calculate_direction(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> float:
        """
        Calculate direction in degrees using standard coordinate system:
        - 0° = East (facing right, positive X/col direction)
        - 90° = North (facing up, negative Y/row direction)
        - 180° = West (facing left, negative X/col direction)
        - 270° = South (facing down, positive Y/row direction)
        """
        dr = to_pos[0] - from_pos[0]  # row difference (positive = moving down/south)
        dc = to_pos[1] - from_pos[1]  # col difference (positive = moving right/east)
        
        # Calculate angle using atan2(y, x) where:
        # - y = -dr (negative because row increase = moving down, but we want up = positive)
        # - x = dc (positive because col increase = moving right)
        angle = math.degrees(math.atan2(-dr, dc))
        
        # Normalize to [0, 360)
        if angle < 0:
            angle += 360
            
        return angle

    def get_direction_name(self, degrees: float) -> str:
        """Convert degrees to human-readable direction name"""
        degrees = degrees % 360
        
        if degrees < 22.5 or degrees >= 337.5:
            return "East"
        elif degrees < 67.5:
            return "Northeast"
        elif degrees < 112.5:
            return "North"
        elif degrees < 157.5:
            return "Northwest"
        elif degrees < 202.5:
            return "West"
        elif degrees < 247.5:
            return "Southwest"
        elif degrees < 292.5:
            return "South"
        else:  # 292.5 <= degrees < 337.5
            return "Southeast"

    def bresenham_line(self, r1: int, c1: int, r2: int, c2: int) -> List[Tuple[int, int]]:
        """Bresenham's line algorithm to get all points along a line"""
        points = []
        dr = abs(r2 - r1)
        dc = abs(c2 - c1)
        r, c = r1, c1
        r_inc = 1 if r1 < r2 else -1
        c_inc = 1 if c1 < c2 else -1
        error = dr - dc

        while True:
            points.append((r, c))
            if r == r2 and c == c2:
                break
            e2 = 2 * error
            if e2 > -dc:
                error -= dc
                r += r_inc
            if e2 < dr:
                error += dr
                c += c_inc

        return points

    def can_move_directly(self, grid: List[List[int]], start: Tuple[int, int], end: Tuple[int, int]) -> bool:
        """Check if we can move directly between two points without hitting obstacles"""
        r1, c1 = start
        r2, c2 = end
        
        # Use Bresenham's line algorithm to check all points along the line
        points = self.bresenham_line(r1, c1, r2, c2)
        
        for point in points:
            if not self.is_valid_position(grid, point, 0):  # Use margin 0 for direct line check
                return False
        return True

    def smooth_path(self, grid: List[List[int]], path: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Smooth the path by removing unnecessary waypoints"""
        if len(path) <= 2:
            return path
            
        smoothed = [path[0]]
        
        i = 0
        while i < len(path) - 1:
            # Look ahead to find the farthest point we can reach directly
            farthest = i + 1
            for j in range(i + 2, len(path)):
                if self.can_move_directly(grid, path[i], path[j]):
                    farthest = j
                else:
                    break
            
            smoothed.append(path[farthest])
            i = farthest
            
        return smoothed

    def find_path_raw(self, grid: List[List[int]], start: Tuple[int, int], goal: Tuple[int, int], margin: int) -> Optional[List[Tuple[int, int]]]:
        """A* pathfinding that returns every step without smoothing"""
        
        if start == goal:
            return [start]

        frontier = []
        heappush(frontier, (0, start))
        came_from = {start: None}
        cost_so_far = {start: 0}

        while frontier:
            current = heappop(frontier)[1]

            if current == goal:
                break

            for next_pos, move_cost in self.get_neighbors_with_margin(grid, current, margin):
                new_cost = cost_so_far[current] + move_cost
                
                if next_pos not in cost_so_far or new_cost < cost_so_far[next_pos]:
                    cost_so_far[next_pos] = new_cost
                    priority = new_cost + self.heuristic(next_pos, goal)
                    heappush(frontier, (priority, next_pos))
                    came_from[next_pos] = current

        # Check if path was found
        if goal not in came_from:
            return None

        # Reconstruct the COMPLETE path - every step
        path = []
        current = goal
        while current is not None:
            path.append(current)
            current = came_from[current]
        path.reverse()
        
        return path

    def find_path_with_smoothing(self, grid: List[List[int]], start: Tuple[int, int], goal: Tuple[int, int], margin: int) -> Optional[List[Tuple[int, int]]]:
        """A* pathfinding algorithm with specified margin and path smoothing"""
        
        print(f"Starting A* pathfinding with margin {margin}")
        print(f"Start: {start}, Goal: {goal}")
        
        # If start and goal are the same
        if start == goal:
            return [start]

        frontier = []
        heappush(frontier, (0, start))
        came_from = {start: None}
        cost_so_far = {start: 0}
        
        explored_count = 0
        max_explorations = 10000  # Prevent infinite loops

        while frontier and explored_count < max_explorations:
            current = heappop(frontier)[1]
            explored_count += 1
            
            if explored_count % 1000 == 0:
                print(f"Explored {explored_count} nodes, current: {current}")

            if current == goal:
                print(f"Found goal! Explored {explored_count} nodes")
                break

            neighbors = self.get_neighbors_with_margin(grid, current, margin)
            if explored_count < 5:  # Debug first few iterations
                print(f"Node {current} has {len(neighbors)} valid neighbors: {[n[0] for n in neighbors[:5]]}")
                
            for next_pos, move_cost in neighbors:
                new_cost = cost_so_far[current] + move_cost
                
                if next_pos not in cost_so_far or new_cost < cost_so_far[next_pos]:
                    cost_so_far[next_pos] = new_cost
                    priority = new_cost + self.heuristic(next_pos, goal)
                    heappush(frontier, (priority, next_pos))
                    came_from[next_pos] = current

        # Check if path was found
        if goal not in came_from:
            print(f"Goal not reached after exploring {explored_count} nodes")
            return None

        print(f"Successfully found path! Reconstructing...")
        
        # Reconstruct path
        path = []
        current = goal
        while current is not None:
            path.append(current)
            current = came_from[current]
        path.reverse()
        
        print(f"Raw path has {len(path)} points")
        smoothed_path = self.smooth_path(grid, path)
        print(f"Smoothed path has {len(smoothed_path)} points")
        
        return smoothed_path

    def find_path(self, grid: List[List[int]], start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """A* pathfinding algorithm with adaptive margin consideration and path smoothing"""
        
        start_best_margin = self.find_best_valid_margin(grid, start)
        goal_best_margin = self.find_best_valid_margin(grid, goal)
        
        print(f"Best margins: start={start_best_margin}, goal={goal_best_margin}")
        
        if start_best_margin < 0 or goal_best_margin < 0:
            print(f"Invalid positions: start_margin={start_best_margin}, goal_margin={goal_best_margin}")
            return None
        
        # Try different margins with smoothing
        margin_attempts = []
        if start_best_margin >= 3 and goal_best_margin >= 3:
            margin_attempts.append(3)  # Ideal case
        if start_best_margin >= 2 and goal_best_margin >= 2:
            margin_attempts.append(2)  # Good case
        if start_best_margin >= 1 and goal_best_margin >= 1:
            margin_attempts.append(1)  # Acceptable case
        margin_attempts.append(0)  # Last resort
        
        # Remove duplicates while preserving order
        margin_attempts = list(dict.fromkeys(margin_attempts))
        
        print(f"Trying margins in order: {margin_attempts}")
        
        # Try pathfinding with different margins
        for margin in margin_attempts:
            print(f"Attempting pathfinding with margin {margin}")
            
            # First, do a quick connectivity check
            start_neighbors = self.get_neighbors_with_margin(grid, start, margin)
            goal_neighbors = self.get_neighbors_with_margin(grid, goal, margin)
            
            print(f"  Start position has {len(start_neighbors)} neighbors")
            print(f"  Goal position has {len(goal_neighbors)} neighbors")
            
            if len(start_neighbors) == 0:
                print(f"  SKIP: Start has no valid neighbors with margin {margin}")
                continue
                
            if len(goal_neighbors) == 0:
                print(f"  SKIP: Goal has no valid neighbors with margin {margin}")
                continue
            
            path = self.find_path_with_smoothing(grid, start, goal, margin)
            if path:
                print(f"SUCCESS: Found path with margin {margin}")
                return path
            else:
                print(f"No path found with margin {margin}")
        
        print("No path found with any margin")
        
        # Final attempt: try direct line pathfinding as last resort
        print("Trying direct line as last resort...")
        direct_path = self.bresenham_line(start[0], start[1], goal[0], goal[1])
        
        # Check if direct path is valid (even if it violates margins)
        valid_direct = True
        for point in direct_path:
            if (not (0 <= point[0] < len(grid) and 0 <= point[1] < len(grid[0])) or 
                grid[point[0]][point[1]] == 1):
                valid_direct = False
                break
        
        if valid_direct:
            print(f"Direct path is valid! Using {len(direct_path)} points")
            return direct_path
        else:
            print("Direct path also blocked by obstacles")
        
        return None

    def find_path_detailed(self, grid: List[List[int]], start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """A* pathfinding that returns the complete unsmoothed path for detailed navigation"""
        
        # Find the best valid margins for start and goal
        start_best_margin = self.find_best_valid_margin(grid, start)
        goal_best_margin = self.find_best_valid_margin(grid, goal)
        
        print(f"Best margins for detailed path: start={start_best_margin}, goal={goal_best_margin}")
        
        # If either position is completely invalid
        if start_best_margin < 0 or goal_best_margin < 0:
            print(f"Invalid positions: start_margin={start_best_margin}, goal_margin={goal_best_margin}")
            return None
        
        # Try different margins
        margin_attempts = []
        if start_best_margin >= 2 and goal_best_margin >= 2:
            margin_attempts.append(2)
        if start_best_margin >= 1 and goal_best_margin >= 1:
            margin_attempts.append(1)
        margin_attempts.append(0)  # Last resort
        
        # Remove duplicates
        margin_attempts = list(dict.fromkeys(margin_attempts))
        
        print(f"Trying margins for detailed path: {margin_attempts}")
        
        for margin in margin_attempts:
            print(f"Attempting detailed A* with margin {margin}")
            path = self.find_path_raw(grid, start, goal, margin)
            if path:
                print(f"SUCCESS: Found detailed path with {len(path)} steps using margin {margin}")
                return path
            else:
                print(f"No detailed path found with margin {margin}")
        
        return None

    def get_complete_path(self, grid: List[List[int]], start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get the complete detailed path without smoothing - every grid cell the agent passes through"""
        
        # Find raw A* path without smoothing
        path = self.find_path_detailed(grid, start, goal)
        
        if not path:
            return []
        
        print(f"Complete detailed path has {len(path)} grid cells")
        
        # Validate every step is walkable
        invalid_steps = []
        for i, point in enumerate(path):
            row, col = point
            if (not (0 <= row < len(grid) and 0 <= col < len(grid[0])) or 
                grid[row][col] == 1):  # 1 = obstacle
                invalid_steps.append((i, point, grid[row][col] if 0 <= row < len(grid) and 0 <= col < len(grid[0]) else "OOB"))
        
        if invalid_steps:
            print(f"ERROR: Path goes through {len(invalid_steps)} obstacles!")
            for step, point, value in invalid_steps:
                print(f"  Step {step}: {point} = {value} (should be 0 for walkable)")
            return []
        
        return path

    def calculate_smart_actions(self, path: List[Tuple[int, int]]) -> Tuple[List[dict], float]:
        """Convert path into human-friendly navigation actions"""
        if len(path) <= 1:
            return [], 0.0

        print(f"\nCALCULATING SMART ACTIONS:")
        print(f"Path has {len(path)} points")
        
        # Calculate initial direction from first two points
        if len(path) >= 2:
            initial_direction = self.calculate_direction(path[0], path[1])
            print(f"Initial direction: {initial_direction}° ({self.get_direction_name(initial_direction)})")
        else:
            initial_direction = 0.0
        
        # Group consecutive movements by direction
        segments = []
        i = 0
        
        while i < len(path) - 1:
            current_pos = path[i]
            next_pos = path[i + 1]
            
            # Calculate direction for this segment
            segment_direction = self.calculate_direction(current_pos, next_pos)
            segment_points = [current_pos]
            segment_end = i
            
            # Extend segment while direction remains similar (within 22.5° tolerance)
            while segment_end < len(path) - 1:
                curr_point = path[segment_end]
                next_point = path[segment_end + 1]
                
                point_direction = self.calculate_direction(curr_point, next_point)
                direction_diff = abs(point_direction - segment_direction)
                
                # Handle wrap-around
                if direction_diff > 180:
                    direction_diff = 360 - direction_diff
                
                if direction_diff <= 22.5:  # Same direction
                    segment_points.append(next_point)
                    segment_end += 1
                else:
                    break
            
            # Calculate segment distance
            segment_distance = 0
            for j in range(len(segment_points) - 1):
                curr = segment_points[j]
                next_pt = segment_points[j + 1]
                segment_distance += math.sqrt((next_pt[0] - curr[0])**2 + (next_pt[1] - curr[1])**2)
            
            segments.append({
                'direction': segment_direction,
                'distance': segment_distance,
                'grid_steps': len(segment_points) - 1,
                'start_idx': i,
                'end_idx': segment_end,
                'points': segment_points
            })
            
            i = segment_end
        
        print(f"Created {len(segments)} direction segments:")
        for idx, segment in enumerate(segments):
            direction_name = self.get_direction_name(segment['direction'])
            print(f"  Segment {idx}: {segment['grid_steps']} grid steps {direction_name} ({segment['direction']:.0f}°)")
        
        # Convert segments to actions
        actions = []
        current_facing = initial_direction
        
        for segment in segments:
            target_direction = segment['direction']
            
            # Calculate rotation needed
            rotation_needed = target_direction - current_facing
            
            # Normalize to [-180, 180] - shortest rotation
            while rotation_needed > 180:
                rotation_needed -= 360
            while rotation_needed <= -180:
                rotation_needed += 360
            
            print(f"Rotation calculation: current={current_facing:.1f}° -> target={target_direction:.1f}° = {rotation_needed:.1f}°")
            
            # Add rotation if significant (> 22.5 degrees)
            if abs(rotation_needed) > 22.5:
                # CORRECTED LOGIC: In compass navigation, increasing angle = turning LEFT
                # positive rotation_needed means turning LEFT (counter-clockwise)
                # negative rotation_needed means turning RIGHT (clockwise)
                direction_word = "left" if rotation_needed > 0 else "right"
                actions.append({
                    "type": "rotate",
                    "degrees": abs(int(round(rotation_needed))),
                    "message": f"Rotate {abs(int(round(rotation_needed)))}° {direction_word}"
                })
                current_facing = target_direction
                print(f"Added rotation: {abs(rotation_needed):.0f}° {direction_word}")
                print(f"Rotation logic: {current_facing:.1f}° -> {target_direction:.1f}° = {rotation_needed:.1f}° ({'LEFT (increasing angle)' if rotation_needed > 0 else 'RIGHT (decreasing angle)'})")
            
            # Add movement with simplified message
            user_steps = max(1, round(segment['distance'] / 2))
            
            actions.append({
                "type": "move",
                "steps": user_steps,
                "message": f"Move forward {user_steps} step{'s' if user_steps != 1 else ''}"
            })
            print(f"Added movement: {user_steps} steps forward")
        
        return actions, initial_direction

    def validate_grid_interpretation(self, grid: List[List[int]]) -> None:
        """Validate that we're interpreting the grid correctly (1=obstacle, 0=walkable)"""
        if not grid or not grid[0]:
            return
            
        total_cells = len(grid) * len(grid[0])
        obstacle_count = sum(sum(1 for cell in row if cell == 1) for row in grid)
        walkable_count = sum(sum(1 for cell in row if cell == 0) for row in grid)
        other_count = total_cells - obstacle_count - walkable_count
        
        print(f"\nGRID INTERPRETATION VALIDATION:")
        print(f"Total cells: {total_cells}")
        print(f"Obstacles (1s): {obstacle_count} ({obstacle_count/total_cells*100:.1f}%)")
        print(f"Walkable (0s): {walkable_count} ({walkable_count/total_cells*100:.1f}%)")
        if other_count > 0:
            print(f"Other values: {other_count} ({other_count/total_cells*100:.1f}%)")
        
        # Sanity check: if >80% is obstacles, something might be wrong
        if obstacle_count / total_cells > 0.8:
            print(f"WARNING: {obstacle_count/total_cells*100:.1f}% obstacles seems very high!")
            print("Are you sure 1=obstacle and 0=walkable? Might be inverted.")
        
        if walkable_count / total_cells > 0.95:
            print(f"WARNING: {walkable_count/total_cells*100:.1f}% walkable seems very high!")
            print("This might be mostly empty space.")

    def get_navigation_actions_with_path(self, floor_id: str, poi_id: str, start_pos: Tuple[int, int]) -> Tuple[List[dict], List[Tuple[int, int]], float]:
        """Get navigation actions and the complete detailed path (not smoothed)"""
        try:
            print(f"\n=== NAVIGATION DEBUG ===")
            print(f"Floor ID: {floor_id}")
            print(f"POI ID: {poi_id}")
            print(f"Start position: {start_pos}")
            
            # Get floor grid and POI position
            grid, grid_dimensions = self.get_floor_grid(floor_id)
            goal_pos = self.get_poi_position(poi_id)
            
            print(f"\nGRID INFO:")
            print(f"Grid dimensions: {grid_dimensions}")
            print(f"Grid size: {len(grid)} x {len(grid[0]) if grid else 0}")
            
            # Validate grid interpretation
            self.validate_grid_interpretation(grid)
            
            # Find SMOOTHED path for actions (human-friendly navigation)
            smoothed_path = self.find_path(grid, start_pos, goal_pos)
            
            # Find COMPLETE path for visualization (all steps)
            complete_path = self.get_complete_path(grid, start_pos, goal_pos)
            
            if not smoothed_path or not complete_path:
                print("ERROR: No path found!")
                return [], [], 0.0
            
            print(f"Smoothed path found with {len(smoothed_path)} waypoints")
            print(f"Complete path found with {len(complete_path)} steps")
            
            # Convert SMOOTHED path to actions (for human instructions)
            actions, initial_orientation = self.calculate_smart_actions(smoothed_path)
            print(f"Generated {len(actions)} human-friendly actions")
            print(f"Initial orientation: {initial_orientation:.0f}° ({self.get_direction_name(initial_orientation)})")
            
            print(f"Actions: {actions}")
            print(f"=== END DEBUG ===\n")
            
            # Return actions based on smoothed path, but complete path for visualization
            return actions, complete_path, initial_orientation
            
        except Exception as e:
            print(f"ERROR in navigation: {str(e)}")
            import traceback
            traceback.print_exc()
            return [], [], 0.0

    def get_navigation_actions(self, floor_id: str, poi_id: str, start_pos: Tuple[int, int]) -> List[dict]:
        """Main function to get navigation actions from start position to POI"""
        actions, _, _ = self.get_navigation_actions_with_path(floor_id, poi_id, start_pos)
        return actions