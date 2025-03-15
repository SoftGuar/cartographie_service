import json
import base64
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

# Test data with unique ID to avoid conflicts
test_room_data = {
    "id": f"test-room-{__import__('uuid').uuid4()}",
    "company_id": "company-1",
    "name": "Test Room",
    "building": "Test Building",
    "floor": 1,
    "width": 10.5,
    "height": 8.5,
    "coordinates": [36.7525, 3.0420],
    "grid_data": [[0, 1], [1, 0]],
    "grid_dimensions": [2, 2],
    "image_data": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
}

def test_get_rooms():
    """Test getting rooms - should return a list (empty or not)"""
    response = client.get("/rooms/")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_create_and_get_room():
    """Test creating a room and then retrieving it"""
    # Create room
    response = client.post("/rooms/", json=test_room_data)
    assert response.status_code == 200
    created_room = response.json()
    assert created_room["name"] == test_room_data["name"]
    assert created_room["building"] == test_room_data["building"]
    
    # Get the created room
    response = client.get(f"/rooms/{test_room_data['id']}")
    assert response.status_code == 200
    room = response.json()
    assert room["id"] == test_room_data["id"]
    assert room["name"] == test_room_data["name"]

def test_get_nonexistent_room():
    """Test getting a room that doesn't exist"""
    response = client.get("/rooms/nonexistent-id-12345")
    assert response.status_code == 404

def test_update_room():
    """Test updating a room's grid data"""
    # First create a room
    create_response = client.post("/rooms/", json=test_room_data)
    assert create_response.status_code == 200
    
    # Update data
    update_data = {
        "grid_data": [[1, 1], [1, 1]],
        "grid_dimensions": [2, 2],
        "image_data": None
    }
    
    response = client.put(f"/rooms/{test_room_data['id']}", json=update_data)
    assert response.status_code == 200
    updated_room = response.json()
    assert updated_room["grid_data"] == update_data["grid_data"]
    assert updated_room["grid_dimensions"] == update_data["grid_dimensions"]

def test_room_image():
    """Test room image endpoints"""
    # Create room with image
    response = client.post("/rooms/", json=test_room_data)
    assert response.status_code == 200
    room_id = response.json()["id"]
    
    # Get image
    response = client.get(f"/rooms/{room_id}/image")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    
    # Get nonexistent image
    response = client.get("/rooms/nonexistent-id-12345/image")
    assert response.status_code == 404