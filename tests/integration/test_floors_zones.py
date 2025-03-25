import pytest
from fastapi.testclient import TestClient
from main import app
import json
import uuid

client = TestClient(app)

# --------------------------
# Test Data
# --------------------------

@pytest.fixture
def sample_floor_data():
    return {
        "name": "Test Floor",
        "level": 1,
        "width": 100.0,
        "height": 80.0,
        "coordinates": json.dumps({"points": [[0,0], [100,0], [100,80], [0,80]]}),
        "environment_id": "env-123",
        "grid_data": [[0, 1], [1, 0]],
        "grid_dimensions": [10, 10],
        "image_data": None
    }

@pytest.fixture
def sample_zone_data(sample_floor_data):
    floor_response = client.post("/floors/", json=sample_floor_data)
    assert floor_response.status_code == 200
    return {
        "name": "Test Zone",
        "color": "#FF0000",
        "type_id": "zone-type-123",
        "floor_id": floor_response.json()["id"],
        "shape": [
            {
                "type": "polygon",
                "coordinates": [[[10,10], [50,10], [50,50], [10,50]]]
            }
        ]
    }

# --------------------------
# CRUD Tests
# --------------------------

@pytest.mark.integration
def test_create_zone_with_multiple_shapes(sample_zone_data):
    """Test creating a zone with multiple shapes"""
    zone_data = {**sample_zone_data}
    zone_data["shape"].append({
        "type": "circle",
        "center": [30,30],
        "radius": 10
    })
    
    response = client.post("/zones/", json=zone_data)
    assert response.status_code == 200
    result = response.json()
    assert len(result["shape"]) == 2

@pytest.mark.integration
def test_get_zone_with_shapes(sample_zone_data):
    """Test retrieving a zone with its shapes"""
    create_response = client.post("/zones/", json=sample_zone_data)
    zone_id = create_response.json()["id"]
    
    response = client.get(f"/zones/{zone_id}")
    assert response.status_code == 200
    assert response.json()["id"] == zone_id

@pytest.mark.integration
def test_update_zone_shapes(sample_zone_data):
    """Test updating a zone's shapes"""
    create_response = client.post("/zones/", json=sample_zone_data)
    zone_id = create_response.json()["id"]
    
    update_data = {
        "name": "Updated Zone",
        "type_id": "zone-type-123",
        "floor_id": sample_zone_data["floor_id"],
        "color": "#00FF00",
        "shape": [
            {
                "type": "rectangle",
                "coordinates": [[5,5], [15,15]]
            }
        ]
    }
    
    response = client.put(f"/zones/{zone_id}", json=update_data)
    assert response.status_code == 200
    updated = response.json()
    assert updated["shape"][0]["type"] == "rectangle"

# --------------------------
# Validation Tests
# --------------------------

@pytest.mark.integration
def test_invalid_shape_format(sample_zone_data):
    """Test invalid shape structure"""
    invalid_data = {**sample_zone_data}
    invalid_data["shape"] = [{
        "type": "polygon", 
        "coordinates": "invalid_string"  # Should be List[List[List[float]]]
    }]
    
    response = client.post("/zones/", json=invalid_data)
    assert response.status_code == 422
    assert any(err["type"] == "list_type" for err in response.json()["detail"])

@pytest.mark.integration
def test_empty_shapes_list(sample_zone_data):
    """Test zone with no shapes should fail validation"""
    invalid_data = {**sample_zone_data}
    invalid_data["shape"] = []
    
    response = client.post("/zones/", json=invalid_data)
    
    if response.status_code == 200:
        pytest.fail("Empty shapes list should fail validation but returned 200")
    assert response.status_code == 422
    assert "at least one shape" in response.text.lower()

# --------------------------
# Relationship Tests
# --------------------------

@pytest.mark.integration
def test_zone_floor_relationship(sample_floor_data, sample_zone_data):
    """Test zone is properly associated with floor"""
    # Create zone
    zone_response = client.post("/zones/", json=sample_zone_data)
    zone_id = zone_response.json()["id"]
    floor_id = sample_zone_data["floor_id"]
    
    # Verify relationship in zone response
    assert zone_response.json()["floor_id"] == floor_id
    
    # Verify floor contains zone reference
    floor_response = client.get(f"/floors/{floor_id}")
    floor_data = floor_response.json()
    
    # Debug print to inspect the actual response
    print("\nActual floor response:", floor_data)
    
    # Check if zones are being included in the response
    if "zones" not in floor_data:
        # If zones are missing, check the response model
        pytest.fail("'zones' field missing from floor response. "
                   "Ensure your FloorResponse schema includes: "
                   "zones: List[ZoneResponse] = Field(default_factory=list)")
    
    assert isinstance(floor_data["zones"], list)
    assert any(zone["id"] == zone_id for zone in floor_data["zones"])

# --------------------------
# Edge Case Tests
# --------------------------

@pytest.mark.integration
def test_create_zone_with_invalid_color(sample_zone_data):
    """Test zone creation with invalid color format"""
    invalid_data = {**sample_zone_data}
    invalid_data["color"] = "red"  # Invalid hex format
    
    response = client.post("/zones/", json=invalid_data)
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert any("string_pattern_mismatch" in err["type"] for err in detail)