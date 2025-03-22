import pytest
from unittest.mock import MagicMock
from models.zone import Zone
from schemas.zone import ZoneCreate, ZoneUpdate
from services.zone_service import create_zone, get_zone, update_zone, delete_zone, get_zones_by_floor

def test_create_zone(mock_db):
    """Test creating a zone."""
    # Mock data
    zone_data = ZoneCreate(
        name="Test Zone",
        color="#FFFFFF",
        type_id="type-123",
        shape={"type": "polygon", "coordinates": [[0, 0], [1, 1], [1, 0]]},
        floor_id="floor-123",
    )

    # Mock the database session
    mock_db.add = MagicMock()
    mock_db.commit = MagicMock()
    mock_db.refresh = MagicMock()

    # Call the service function
    result = create_zone(mock_db, zone_data)

    # Assertions
    assert isinstance(result, Zone)
    assert result.name == zone_data.name
    assert result.color == zone_data.color
    assert result.type_id == zone_data.type_id
    assert result.shape == zone_data.shape
    assert result.floor_id == zone_data.floor_id
    mock_db.add.assert_called_once()
    mock_db.commit.assert_called_once()
    mock_db.refresh.assert_called_once()

def test_get_zone(mock_db):
    """Test retrieving a zone by ID."""
    # Mock data
    zone_id = "zone-123"
    mock_zone = Zone(id=zone_id, name="Test Zone")

    # Mock the database query
    mock_db.query.return_value.filter.return_value.first.return_value = mock_zone

    # Call the service function
    result = get_zone(mock_db, zone_id)

    # Assertions
    assert result == mock_zone
    mock_db.query.return_value.filter.return_value.first.assert_called_once()

def test_update_zone(mock_db):
    """Test updating a zone."""
    # Mock data
    zone_id = "zone-123"
    update_data = ZoneUpdate(name="Updated Zone")
    mock_zone = Zone(id=zone_id, name="Test Zone")

    # Mock the database query
    mock_db.query.return_value.filter.return_value.first.return_value = mock_zone
    mock_db.commit = MagicMock()
    mock_db.refresh = MagicMock()

    # Call the service function
    result = update_zone(mock_db, zone_id, update_data)

    # Assertions
    assert result.name == update_data.name
    mock_db.commit.assert_called_once()
    mock_db.refresh.assert_called_once()

def test_delete_zone(mock_db):
    """Test deleting a zone."""
    # Mock data
    zone_id = "zone-123"
    mock_zone = Zone(id=zone_id, name="Test Zone")

    # Mock the database query
    mock_db.query.return_value.filter.return_value.first.return_value = mock_zone
    mock_db.delete = MagicMock()
    mock_db.commit = MagicMock()

    # Call the service function
    result = delete_zone(mock_db, zone_id)

    # Assertions
    assert result is True
    mock_db.delete.assert_called_once_with(mock_zone)
    mock_db.commit.assert_called_once()

def test_get_zones_by_floor(mock_db):
    """Test retrieving zones by floor ID."""
    # Mock data
    floor_id = "floor-123"
    mock_zones = [
        Zone(id="zone-1", name="Zone 1", floor_id=floor_id),
        Zone(id="zone-2", name="Zone 2", floor_id=floor_id),
    ]

    # Mock the database query
    mock_db.query.return_value.filter.return_value.all.return_value = mock_zones

    # Call the service function
    result = get_zones_by_floor(mock_db, floor_id)

    # Assertions
    assert result == mock_zones
    mock_db.query.return_value.filter.return_value.all.assert_called_once()