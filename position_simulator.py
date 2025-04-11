import asyncio
import aiohttp
import math
import random
from datetime import datetime
from typing import List, Tuple, Dict

class RoomLocation:
    def __init__(self, name: str, position: Tuple[float, float], stay_duration: Tuple[int, int]):
        self.name = name
        self.position = position
        self.stay_duration = stay_duration  # (min, max) seconds to stay at location

class PositionSimulator:
    def __init__(self):
        self.base_url = "http://localhost:8000/update_position"  # Updated port to 8000
        
        # Define room locations with realistic positions and stay durations
        self.locations = {
            "bed": RoomLocation("bed", (100, 100), (3, 8)),  # Stay 3-8 seconds
            "desk": RoomLocation("desk", (300, 100), (5, 15)),  # Stay 5-15 seconds
            "door": RoomLocation("door", (400, 300), (2, 4)),  # Stay 2-4 seconds
            "wardrobe": RoomLocation("wardrobe", (100, 300), (3, 7)),  # Stay 3-7 seconds
            "window": RoomLocation("window", (400, 100), (2, 5)),  # Stay 2-5 seconds
        }
        
        # Define common paths/routines
        self.routines = [
            # Morning routine
            ["bed", "wardrobe", "desk", "door"],
            # Study routine
            ["door", "desk", "window", "desk", "door"],
            # Evening routine
            ["door", "desk", "wardrobe", "bed"],
        ]
        
        self.current_routine = []
        self.current_position = self.locations["bed"].position
        self.target_position = self.current_position
        self.current_angle = 0
        self.base_speed = 3  # Base pixels per update
        self.current_location = None
        self.location_timer = 0
        
        # Add some randomness to movement
        self.wandering_amplitude = 10  # Maximum pixels to deviate from direct path
        self.wandering_frequency = 0.1  # How often to change wandering direction
        self.wandering_offset = 0
        self.last_wandering_update = 0

    def calculate_angle(self, start: Tuple[float, float], end: Tuple[float, float]) -> float:
        """Calculate angle between two points in degrees"""
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        angle = math.degrees(math.atan2(dy, dx))
        return (angle + 360) % 360

    def add_wandering(self, position: Tuple[float, float], target: Tuple[float, float]) -> Tuple[float, float]:
        """Add natural wandering to movement"""
        current_time = datetime.now().timestamp()
        
        # Update wandering offset periodically
        if current_time - self.last_wandering_update > self.wandering_frequency:
            self.wandering_offset = random.uniform(-self.wandering_amplitude, self.wandering_amplitude)
            self.last_wandering_update = current_time
        
        # Calculate perpendicular vector to movement direction
        dx = target[0] - position[0]
        dy = target[1] - position[1]
        length = math.sqrt(dx * dx + dy * dy)
        
        if length > 0:
            # Add wandering perpendicular to movement direction
            wx = -dy / length * self.wandering_offset
            wy = dx / length * self.wandering_offset
            return (position[0] + wx, position[1] + wy)
        
        return position

    def get_next_position(self) -> Tuple[Tuple[float, float], float]:
        """Calculate next position with realistic movement"""
        current_time = datetime.now().timestamp()
        
        # If at a location, wait for the designated duration
        if self.current_location and self.location_timer > 0:
            if current_time < self.location_timer:
                return self.current_position, self.current_angle
            else:
                self.current_location = None
                self.location_timer = 0
        
        # If no current routine, start a new one
        if not self.current_routine:
            self.current_routine = random.choice(self.routines).copy()
            next_location = self.locations[self.current_routine[0]]
            self.target_position = next_location.position
            print(f"Starting new routine, heading to {next_location.name}")
        
        current_x, current_y = self.current_position
        target_x, target_y = self.target_position
        
        # Add natural wandering to movement
        wandered_position = self.add_wandering((current_x, current_y), (target_x, target_y))
        current_x, current_y = wandered_position
        
        # Calculate direction and distance
        dx = target_x - current_x
        dy = target_y - current_y
        distance = math.sqrt(dx * dx + dy * dy)
        
        # Vary speed based on distance to target
        speed = self.base_speed
        if distance < 50:  # Slow down when approaching target
            speed = max(1, self.base_speed * (distance / 50))
        
        if distance < speed:
            # Reached target location
            self.current_position = (target_x, target_y)
            location_name = self.current_routine.pop(0)
            location = self.locations[location_name]
            
            # Set timer for staying at this location
            self.current_location = location
            stay_time = random.uniform(*location.stay_duration)
            self.location_timer = current_time + stay_time
            print(f"Reached {location.name}, staying for {stay_time:.1f} seconds")
            
            # Set next target if there are more locations in routine
            if self.current_routine:
                next_location = self.locations[self.current_routine[0]]
                self.target_position = next_location.position
                print(f"Next destination: {next_location.name}")
            
            return self.current_position, self.current_angle
        
        # Calculate movement with variable speed
        angle = math.atan2(dy, dx)
        new_x = current_x + speed * math.cos(angle)
        new_y = current_y + speed * math.sin(angle)
        
        # Update position and angle
        self.current_position = (new_x, new_y)
        self.current_angle = math.degrees(angle)
        
        return self.current_position, self.current_angle

    async def send_update(self, session: aiohttp.ClientSession, position: Tuple[float, float], angle: float):
        """Send position update to the server"""
        data = {
            "x": position[0],
            "y": position[1],
            "angle": angle
        }
        try:
            async with session.post(self.base_url, json=data) as response:
                if response.status != 200:
                    print(f"Error sending update: {response.status}")
                await response.text()
        except Exception as e:
            print(f"Error sending update: {e}")

    async def run(self):
        """Main simulation loop"""
        async with aiohttp.ClientSession() as session:
            while True:
                position, angle = self.get_next_position()
                await self.send_update(session, position, angle)
                
                # Variable delay based on movement state
                if self.current_location:
                    delay = random.uniform(0.2, 0.4)  # Slower updates when stationary
                else:
                    delay = random.uniform(0.05, 0.15)  # Faster updates when moving
                
                await asyncio.sleep(delay)

async def main():
    simulator = PositionSimulator()
    await simulator.run()

if __name__ == "__main__":
    print("Starting position simulator...")
    print("Press Ctrl+C to stop")
    asyncio.run(main())