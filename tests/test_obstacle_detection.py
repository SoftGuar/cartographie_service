import io
import json
import base64
import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

@pytest.fixture
def sample_floorplan_image():
    """Create a simple test image"""
    return "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="

@pytest.fixture
def sample_image_data():
    """Create sample image data for testing"""
    return base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==")

@pytest.fixture
def sample_files(sample_image_data):
    """Create sample files for testing"""
    return {"file": ("test.png", io.BytesIO(sample_image_data), "image/png")}

@pytest.mark.integration
def test_process_floor_plan_basic(sample_files):
    """Test basic floor plan processing without any options"""
    response = client.post("/process_floor_plan", files=sample_files)
    assert response.status_code == 200
    
    result = response.json()
    assert "grid" in result
    assert "grid_dimensions" in result
    assert isinstance(result["grid"], list)
    assert isinstance(result["grid_dimensions"], list)
    assert len(result["grid_dimensions"]) == 2


@pytest.mark.unit
def test_process_floor_plan_invalid_image():
    """Test error handling for invalid image input"""
    invalid_data = b"not an image"
    files = {"file": ("test.png", io.BytesIO(invalid_data), "image/png")}
    
    response = client.post("/process_floor_plan", files=files)
    assert response.status_code == 500
    assert "Error processing floor plan" in response.json()["detail"]


@pytest.mark.integration
def test_process_floor_plan_text_removal(sample_files):
    """Test text removal functionality specifically"""
    options = {
        "grid_size": 4,
        "include_text_removal": True,
        "include_walls_detection": False,
        "include_furniture_detection": False,
        "include_doors_detection": False
    }
    
    response = client.post(
        "/process_floor_plan",
        files=sample_files,
        data={"options": json.dumps(options)}
    )
    
    assert response.status_code == 200
    result = response.json()
    assert "no_text_image" in result
    assert result["no_text_image"] is not None


@pytest.mark.integration
def test_process_floor_plan_walls_only(sample_files):
    """Test walls detection functionality specifically"""
    options = {
        "grid_size": 4,
        "include_text_removal": False,
        "include_walls_detection": True,
        "include_furniture_detection": False,
        "include_doors_detection": False
    }
    
    response = client.post(
        "/process_floor_plan",
        files=sample_files,
        data={"options": json.dumps(options)}
    )
    
    assert response.status_code == 200
    result = response.json()
    assert "walls_only" in result
    assert result["walls_only"] is not None
    assert "grid" in result
    assert isinstance(result["grid"], list)


@pytest.mark.integration
def test_process_floor_plan_furniture_only(sample_files):
    """Test furniture detection functionality specifically"""
    options = {
        "grid_size": 4,
        "include_text_removal": False,
        "include_walls_detection": False,
        "include_furniture_detection": True,
        "include_doors_detection": False
    }
    
    response = client.post(
        "/process_floor_plan",
        files=sample_files,
        data={"options": json.dumps(options)}
    )
    
    assert response.status_code == 200
    result = response.json()
    assert "black_furniture" in result
    assert result["black_furniture"] is not None
    assert "grid" in result
    assert isinstance(result["grid"], list)