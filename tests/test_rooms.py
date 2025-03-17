import json
import base64
import uuid
import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

# Track rooms created during tests
test_room_ids = set()

@pytest.fixture
def cleanup_test_rooms():
    """Cleanup only rooms created during tests"""
    yield
    # After each test, clean up only the rooms we created
    for room_id in test_room_ids:
        try:
            client.delete(f"/rooms/{room_id}")
        except:
            pass  # Ignore errors during cleanup
    test_room_ids.clear()

@pytest.fixture
def valid_room_data():
    """Create valid test room data"""
    room_id = f"test-room-{uuid.uuid4()}"
    test_room_ids.add(room_id)  # Track this room for cleanup
    return {
        "id": room_id,
        "company_id": "test-company",  # Use a specific company ID for test data
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
    """Test getting rooms - should return existing rooms"""
    response = client.get("/rooms/")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_create_and_get_room(valid_room_data, cleanup_test_rooms):
    """Test creating a room and then retrieving it"""
    # Create room
    response = client.post("/rooms/", json=valid_room_data)
    assert response.status_code == 200
    created_room = response.json()
    assert created_room["name"] == valid_room_data["name"]
    assert created_room["building"] == valid_room_data["building"]
    
    # Get the created room
    response = client.get(f"/rooms/{valid_room_data['id']}")
    assert response.status_code == 200
    room = response.json()
    assert room["id"] == valid_room_data["id"]
    assert room["name"] == valid_room_data["name"]

def test_get_nonexistent_room():
    """Test getting a room that doesn't exist"""
    response = client.get("/rooms/nonexistent-id-12345")
    assert response.status_code == 404

def test_update_room(valid_room_data, cleanup_test_rooms):
    """Test updating a room's grid data"""
    # First create a room
    create_response = client.post("/rooms/", json=valid_room_data)
    assert create_response.status_code == 200
    
    # Update data
    update_data = {
        "grid_data": [[1, 1], [1, 1]],
        "grid_dimensions": [2, 2],
        "image_data": None
    }
    
    response = client.put(f"/rooms/{valid_room_data['id']}", json=update_data)
    assert response.status_code == 200
    updated_room = response.json()
    assert updated_room["grid_data"] == update_data["grid_data"]
    assert updated_room["grid_dimensions"] == update_data["grid_dimensions"]

def test_get_multiple_rooms(valid_room_data, cleanup_test_rooms):
    """Test creating and retrieving multiple test rooms"""
    # Get initial count of test company rooms
    initial_response = client.get("/rooms/")
    initial_test_rooms = [r for r in initial_response.json() if r["company_id"] == "test-company"]
    initial_count = len(initial_test_rooms)
    
    # Create multiple rooms
    num_rooms = 3
    created_rooms = []
    for i in range(num_rooms):
        room_data = valid_room_data.copy()
        room_id = f"test-room-{uuid.uuid4()}"
        test_room_ids.add(room_id)  # Track for cleanup
        room_data["id"] = room_id
        room_data["name"] = f"Test Room {i}"
        response = client.post("/rooms/", json=room_data)
        assert response.status_code == 200
        created_rooms.append(response.json())
    
    # Get all rooms and filter by test company
    response = client.get("/rooms/")
    assert response.status_code == 200
    test_rooms = [r for r in response.json() if r["company_id"] == "test-company"]
    assert len(test_rooms) == initial_count + num_rooms

def test_room_image(valid_room_data, cleanup_test_rooms):
    """Test room image endpoints"""
    # Create room with image
    response = client.post("/rooms/", json=valid_room_data)
    assert response.status_code == 200
    room_id = response.json()["id"]
    
    # Get image
    response = client.get(f"/rooms/{room_id}/image")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    
    # Get nonexistent image
    response = client.get("/rooms/nonexistent-id-12345/image")
    assert response.status_code == 404

def test_create_room_invalid_data(cleanup_test_rooms):
    """Test creating a room with invalid data"""
    invalid_data = {
        "id": f"test-room-{uuid.uuid4()}",
        "name": "Invalid Room",
        # Missing required fields
    }
    test_room_ids.add(invalid_data["id"])
    response = client.post("/rooms/", json=invalid_data)
    assert response.status_code == 422  # Validation error

def test_create_room_duplicate_id(valid_room_data, cleanup_test_rooms):
    """Test creating a room with duplicate ID"""
    # Create first room
    response = client.post("/rooms/", json=valid_room_data)
    assert response.status_code == 200
    
    # Try to create another room with same ID
    response = client.post("/rooms/", json=valid_room_data)
    assert response.status_code == 200  # Should update existing room
    
def test_update_nonexistent_room():
    """Test updating a room that doesn't exist"""
    update_data = {
        "grid_data": [[1, 1], [1, 1]],
        "grid_dimensions": [2, 2],
        "image_data": None
    }
    response = client.put("/rooms/nonexistent-id-12345", json=update_data)
    assert response.status_code == 404

def test_get_room_image_invalid_base64(valid_room_data, cleanup_test_rooms):
    """Test handling invalid base64 image data"""
    # Create room with invalid base64 image data
    invalid_data = valid_room_data.copy()
    invalid_data["id"] = f"test-room-{uuid.uuid4()}"
    test_room_ids.add(invalid_data["id"])
    invalid_data["image_data"] = "data:image/png;base64,invalid_base64"
    
    response = client.post("/rooms/", json=invalid_data)
    assert response.status_code == 400
    assert "Invalid base64 image data" in response.json()["detail"]

def test_update_room_invalid_base64(valid_room_data, cleanup_test_rooms):
    """Test updating a room with invalid base64 image data"""
    # First create a valid room
    response = client.post("/rooms/", json=valid_room_data)
    assert response.status_code == 200
    
    # Try to update with invalid base64
    update_data = {
        "grid_data": [[1, 1], [1, 1]],
        "grid_dimensions": [2, 2],
        "image_data": "data:image/png;base64,invalid_base64"
    }
    
    response = client.put(f"/rooms/{valid_room_data['id']}", json=update_data)
    assert response.status_code == 400
    assert "Invalid base64 image data" in response.json()["detail"]
