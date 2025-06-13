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
        # So grid_dimensions_raw[0] = cols, grid_dimensions_raw[1] = rows
        actual_rows = len(grid_data)
        actual_cols = len(grid_data[0]) if grid_data else 0
        db_cols, db_rows = grid_dimensions_raw[0], grid_dimensions_raw[1]
        
        print(f"Grid dimensions analysis:")
        print(f"  Database stored: [cols={db_cols}, rows={db_rows}]")
        print(f"  Actual grid size: {actual_rows} rows x {actual_cols} cols")
        print(f"  Returning: (rows={actual_rows}, cols={actual_cols})")
        
        # Verify consistency
        if actual_rows != db_rows or actual_cols != db_cols:
            print(f"WARNING: Dimension mismatch!")
            print(f"  Expected from DB: {db_rows} rows x {db_cols} cols")
            print(f"  Actual in grid: {actual_rows} rows x {actual_cols} cols")
        
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
        # Note: point.y corresponds to row, point.x corresponds to col
        row = int(round(point.y))
        col = int(round(point.x))
        print(f"POI position: point.x={point.x}, point.y={point.y} -> grid(row={row}, col={col})")
        return row, col

    def is_valid_position(self, grid: List[List[int]], pos: Tuple[int, int], margin: int = OBSTACLE_MARGIN) -> bool:
        """
        Check if a position is valid considering the obstacle margin
        NOTE: 1 = obstacle, 0 = walkable (as per user specification)
        """
        rows, cols = len(grid), len(grid[0]) if grid else 0
        row, col = pos
        
        # Check if position is within grid bounds
        if not (0 <= row < rows and 0 <= col < cols):
            return False
            
        # Check if position itself is an obstacle (1 = obstacle, 0 = walkable)
        if grid[row][col] == 1:
            return False
            
        # Check margin around the position for obstacles
        obstacle_count = 0
        for dr in range(-margin, margin + 1):
            for dc in range(-margin, margin + 1):
                check_row, check_col = row + dr, col + dc
                if (0 <= check_row < rows and 0 <= check_col < cols and 
                    grid[check_row][check_col] == 1):  # 1 = obstacle
                    obstacle_count += 1
        
        if obstacle_count > 0:
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
        """Calculate Euclidean distance between two points (more accurate than Manhattan)"""
        return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

    def find_best_valid_margin(self, grid: List[List[int]], pos: Tuple[int, int], max_margin: int = 5) -> int:
        """Find the maximum valid margin for a position"""
        for margin in range(max_margin, -1, -1):  # Start from max_margin down to 0
            if self.is_valid_position(grid, pos, margin):
                return margin
        return -1  # If even margin 0 fails, return -1

    def find_path(self, grid: List[List[int]], start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """Get complete detailed path through maze - every grid cell"""
        return self.get_complete_path(grid, start, goal)

    def find_path_with_margin(self, grid: List[List[int]], start: Tuple[int, int], goal: Tuple[int, int], margin: int) -> Optional[List[Tuple[int, int]]]:
        """A* pathfinding algorithm with specified margin - returns smoothed path"""
        
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

    def get_complete_path(self, grid: List[List[int]], start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get the complete detailed path without smoothing - every grid cell the agent passes through"""
        
        # Find raw A* path without smoothing
        path = self.find_path_raw(grid, start, goal)
        
        if not path:
            return []
        
        # Don't smooth - return the complete detailed path
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

    def find_path_raw(self, grid: List[List[int]], start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """A* pathfinding that returns the complete unsmoothed path"""
        
        # Find the best valid margins for start and goal
        start_best_margin = self.find_best_valid_margin(grid, start)
        goal_best_margin = self.find_best_valid_margin(grid, goal)
        
        print(f"Best margins: start={start_best_margin}, goal={goal_best_margin}")
        
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
        
        print(f"Trying margins in order: {margin_attempts}")
        
        for margin in margin_attempts:
            print(f"Attempting A* with margin {margin}")
            path = self.find_raw_path_with_margin(grid, start, goal, margin)
            if path:
                print(f"SUCCESS: Found detailed path with {len(path)} steps using margin {margin}")
                return path
            else:
                print(f"No path found with margin {margin}")
        
        return None

    def find_raw_path_with_margin(self, grid: List[List[int]], start: Tuple[int, int], goal: Tuple[int, int], margin: int) -> Optional[List[Tuple[int, int]]]:
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
        
        return path  # Return raw path WITHOUT smoothing

    def can_move_directly(self, grid: List[List[int]], start: Tuple[int, int], end: Tuple[int, int]) -> bool:
        """Check if we can move directly between two points without hitting obstacles"""
        r1, c1 = start
        r2, c2 = end
        
        # Use Bresenham's line algorithm to check all points along the line
        points = self.bresenham_line(r1, c1, r2, c2)
        
        for point in points:
            if not self.is_valid_position(grid, point):
                return False
        return True

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

    def calculate_direction(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> int:
        """
        Calculate direction in degrees where:
        - 0° = North (moving up, decreasing row/y)
        - 90° = East (moving right, increasing col/x) 
        - 180° = South (moving down, increasing row/y)
        - 270° = West (moving left, decreasing col/x)
        
        Forward movement means moving in the direction you're currently facing.
        """
        dr = to_pos[0] - from_pos[0]  # row difference (positive = moving down/south)
        dc = to_pos[1] - from_pos[1]  # col difference (positive = moving right/east)
        
        # Calculate angle in degrees using atan2(y, x) where:
        # - North (up): dr=-1, dc=0 -> atan2(-1, 0) = -90° -> 270°
        # - East (right): dr=0, dc=1 -> atan2(0, 1) = 0° -> 90°
        # - South (down): dr=1, dc=0 -> atan2(1, 0) = 90° -> 180°
        # - West (left): dr=0, dc=-1 -> atan2(0, -1) = 180° -> 270°
        
        angle = math.degrees(math.atan2(dr, dc))
        
        # Convert to compass directions (0=North, 90=East, 180=South, 270=West)
        compass_angle = (angle + 90) % 360
        
        return int(compass_angle)

    def calculate_actions(self, path: List[Tuple[int, int]]) -> List[dict]:
        """
        Convert path into a list of actions (move forward or rotate).
        - 1 step = 2 grid squares
        - Only rotate for changes > 30 degrees
        """
        if len(path) <= 1:
            return []

        actions = []
        current_direction = None
        
        print(f"\nCALCULATING ACTIONS:")
        print(f"Path has {len(path)} points")
        print(f"Total grid distance: {len(path) - 1} squares")
        
        # Calculate total path distance in grid squares first
        total_grid_distance = 0
        for i in range(len(path) - 1):
            curr = path[i]
            next_point = path[i + 1]
            segment_distance = math.sqrt((next_point[0] - curr[0])**2 + (next_point[1] - curr[1])**2)
            total_grid_distance += segment_distance
        
        print(f"Total path distance: {total_grid_distance:.1f} grid squares")
        print(f"Expected steps (distance/2): {total_grid_distance/2:.1f}")
        
        # Group path into direction segments
        segments = []
        i = 0
        
        while i < len(path) - 1:
            current_pos = path[i]
            next_pos = path[i + 1]
            
            # Calculate direction for this segment
            segment_direction = self.calculate_direction(current_pos, next_pos)
            segment_distance = 0
            segment_end = i
            
            # Extend segment as far as possible in the same direction (within 15° tolerance)
            while segment_end < len(path) - 1:
                curr_point = path[segment_end]
                next_point = path[segment_end + 1]
                
                point_direction = self.calculate_direction(curr_point, next_point)
                direction_diff = abs(point_direction - segment_direction)
                
                # Handle wrap-around (e.g., 350° and 10° are close)
                if direction_diff > 180:
                    direction_diff = 360 - direction_diff
                
                if direction_diff <= 15:  # Within tolerance
                    point_distance = math.sqrt((next_point[0] - curr_point[0])**2 + (next_point[1] - curr_point[1])**2)
                    segment_distance += point_distance
                    segment_end += 1
                else:
                    break
            
            segments.append({
                'direction': segment_direction,
                'distance': segment_distance,
                'start_idx': i,
                'end_idx': segment_end
            })
            
            i = segment_end
        
        print(f"Created {len(segments)} segments:")
        for idx, segment in enumerate(segments):
            print(f"  Segment {idx}: {segment['distance']:.1f} squares at {segment['direction']}°")
        
        # Convert segments to actions
        for segment in segments:
            # Handle rotation
            if current_direction is None:
                current_direction = segment['direction']
                print(f"Initial direction: {current_direction}°")
            else:
                rotation_needed = segment['direction'] - current_direction
                
                # Normalize to [-180, 180]
                if rotation_needed > 180:
                    rotation_needed -= 360
                elif rotation_needed <= -180:
                    rotation_needed += 360
                
                if abs(rotation_needed) > 30:
                    direction_word = "right" if rotation_needed > 0 else "left"
                    actions.append({
                        "type": "rotate",
                        "degrees": abs(rotation_needed),
                        "message": f"Rotate {abs(rotation_needed)} degrees {direction_word}"
                    })
                    current_direction = segment['direction']
                    print(f"Added rotation: {abs(rotation_needed)}° {direction_word}")
            
            # Handle movement - convert grid distance to steps (1 step = 2 grid squares)
            steps = max(1, round(segment['distance'] / 2))
            
            actions.append({
                "type": "move",
                "steps": steps,
                "message": f"Move forward {steps} step{'s' if steps > 1 else ''}"
            })
            print(f"Added movement: {steps} steps (from {segment['distance']:.1f} grid squares)")
        
        # Final verification
        total_action_steps = sum(action.get("steps", 0) for action in actions if action.get("type") == "move")
        print(f"Total steps in actions: {total_action_steps}")
        print(f"Original path distance: {total_grid_distance:.1f} squares")
        print(f"Conversion ratio: {total_action_steps / (total_grid_distance/2):.2f} (should be close to 1.0)")
        
        return actions

    def optimize_actions(self, actions: List[dict]) -> List[dict]:
        """
        Optimize actions by combining consecutive moves and rotations.
        - Only combine rotations if total > 30 degrees
        - Properly handle step scaling (1 step = 2 grid squares)
        """
        if not actions:
            return actions
            
        optimized = []
        i = 0
        
        while i < len(actions):
            current_action = actions[i].copy()
            
            # Combine consecutive moves
            if current_action["type"] == "move":
                total_steps = current_action["steps"]
                j = i + 1
                
                # Combine all consecutive moves
                while j < len(actions) and actions[j]["type"] == "move":
                    total_steps += actions[j]["steps"]
                    j += 1
                
                if total_steps > 0:
                    current_action["steps"] = total_steps
                    current_action["message"] = f"Move forward {total_steps} step{'s' if total_steps > 1 else ''}"
                    optimized.append(current_action)
                i = j
            
            # Combine consecutive rotations
            elif current_action["type"] == "rotate":
                total_rotation = current_action["degrees"]
                
                # Get the direction of the first rotation
                if "left" in current_action["message"]:
                    total_rotation = -total_rotation
                
                j = i + 1
                
                # Combine all consecutive rotations
                while j < len(actions) and actions[j]["type"] == "rotate":
                    rotation = actions[j]["degrees"]
                    if "left" in actions[j]["message"]:
                        rotation = -rotation
                    total_rotation += rotation
                    j += 1
                
                # Normalize total rotation to [-180, 180]
                while total_rotation > 180:
                    total_rotation -= 360
                while total_rotation <= -180:
                    total_rotation += 360
                
                # Only add rotation if it's significant (more than 30 degrees)
                if abs(total_rotation) > 30:
                    direction_word = "right" if total_rotation > 0 else "left"
                    current_action["degrees"] = abs(total_rotation)
                    current_action["message"] = f"Rotate {abs(total_rotation)} degrees {direction_word}"
                    optimized.append(current_action)
                    print(f"Optimized rotation: {abs(total_rotation)}° {direction_word}")
                else:
                    print(f"Skipped small rotation: {abs(total_rotation)}°")
                
                i = j
            else:
                optimized.append(current_action)
                i += 1
        
        return optimized

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

    def get_navigation_actions_with_path(self, floor_id: str, poi_id: str, start_pos: Tuple[int, int]) -> Tuple[List[dict], List[Tuple[int, int]]]:
        """Get navigation actions and the complete path"""
        try:
            print(f"\n=== NAVIGATION DEBUG ===")
            print(f"Floor ID: {floor_id}")
            print(f"POI ID: {poi_id}")
            print(f"Start position (input): {start_pos}")
            
            # Get floor grid and POI position
            grid, grid_dimensions = self.get_floor_grid(floor_id)
            goal_pos = self.get_poi_position(poi_id)
            
            print(f"\nGRID INFO:")
            print(f"Grid dimensions: {grid_dimensions}")
            print(f"Grid size: {len(grid)} x {len(grid[0]) if grid else 0}")
            
            # Validate grid interpretation
            self.validate_grid_interpretation(grid)
            
            # Find path with adaptive margin
            print(f"\nPATHFINDING:")
            path = self.find_path(grid, start_pos, goal_pos)
            
            if not path:
                print("ERROR: No path found!")
                return [], []
            
            print(f"Path found with {len(path)} waypoints")
            print(f"Path length in grid squares: {len(path) - 1}")
            
            print(f"\nCOMPLETE PATH (all squares you'll pass through):")
            print(f"Format: (row, col) -> x=col, y=row")
            for i, point in enumerate(path):
                row, col = point
                grid_value = grid[row][col] if 0 <= row < len(grid) and 0 <= col < len(grid[0]) else "OOB"
                symbol = "🚫" if grid_value == 1 else "✅" if grid_value == 0 else "❌"
                if i == 0:
                    print(f"  {i:2d}. START: (row={row}, col={col}) -> x={col}, y={row} | grid_value={grid_value} {symbol}")
                elif i == len(path) - 1:
                    print(f"  {i:2d}. GOAL:  (row={row}, col={col}) -> x={col}, y={row} | grid_value={grid_value} {symbol}")
                else:
                    print(f"  {i:2d}.        (row={row}, col={col}) -> x={col}, y={row} | grid_value={grid_value} {symbol}")
            
            # Convert path to actions
            print(f"\nACTION CALCULATION:")
            actions = self.calculate_actions(path)
            print(f"Generated {len(actions)} actions")
            
            print(f"Actions: {actions}")
            print(f"=== END DEBUG ===\n")
            
            return actions, path
            
        except Exception as e:
            print(f"ERROR in navigation: {str(e)}")
            import traceback
            traceback.print_exc()
            return [], []

    def get_navigation_actions(self, floor_id: str, poi_id: str, start_pos: Tuple[int, int]) -> List[dict]:
        """Main function to get navigation actions from start position to POI"""
        actions, _ = self.get_navigation_actions_with_path(floor_id, poi_id, start_pos)
        return actions