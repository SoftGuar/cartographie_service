import pytest
from unittest.mock import MagicMock
from models.poi import POI
from schemas.poi import POICreate, POIUpdate
from services.poi_service import create_poi, get_poi, update_poi, delete_poi, search_pois

def test_create_poi(mock_db):
    """Test creating a POI."""
    # Mock data
    poi_data = POICreate(
        name="Test POI",
        description="Test Description",
        category_id="cat-123",
        point_id="point-123",
    )

    # Mock the database session
    mock_db.add = MagicMock()
    mock_db.commit = MagicMock()
    mock_db.refresh = MagicMock()

    # Call the service function
    result = create_poi(mock_db, poi_data)

    # Assertions
    assert isinstance(result, POI)
    assert result.name == poi_data.name
    assert result.description == poi_data.description
    assert result.category_id == poi_data.category_id
    assert result.point_id == poi_data.point_id
    mock_db.add.assert_called_once()
    mock_db.commit.assert_called_once()
    mock_db.refresh.assert_called_once()

def test_get_poi(mock_db):
    """Test retrieving a POI by ID."""
    # Mock data
    poi_id = "poi-123"
    mock_poi = POI(id=poi_id, name="Test POI")

    # Mock the database query
    mock_db.query.return_value.filter.return_value.first.return_value = mock_poi

    # Call the service function
    result = get_poi(mock_db, poi_id)

    # Assertions
    assert result == mock_poi
    mock_db.query.return_value.filter.return_value.first.assert_called_once()

def test_update_poi(mock_db):
    """Test updating a POI."""
    # Mock data
    poi_id = "poi-123"
    update_data = POIUpdate(name="Updated POI")
    mock_poi = POI(id=poi_id, name="Test POI")

    # Mock the database query
    mock_db.query.return_value.filter.return_value.first.return_value = mock_poi
    mock_db.commit = MagicMock()
    mock_db.refresh = MagicMock()

    # Call the service function
    result = update_poi(mock_db, poi_id, update_data)

    # Assertions
    assert result.name == update_data.name
    mock_db.commit.assert_called_once()
    mock_db.refresh.assert_called_once()

def test_delete_poi(mock_db):
    """Test deleting a POI."""
    # Mock data
    poi_id = "poi-123"
    mock_poi = POI(id=poi_id, name="Test POI")

    # Mock the database query
    mock_db.query.return_value.filter.return_value.first.return_value = mock_poi
    mock_db.delete = MagicMock()
    mock_db.commit = MagicMock()

    # Call the service function
    result = delete_poi(mock_db, poi_id)

    # Assertions
    assert result is True
    mock_db.delete.assert_called_once_with(mock_poi)
    mock_db.commit.assert_called_once()

def test_search_pois(mock_db):
    """Test searching for POIs by name."""
    # Mock data
    query = "Test"
    mock_pois = [
        POI(id="poi-1", name="Test POI 1"),
        POI(id="poi-2", name="Test POI 2"),
    ]

    # Mock the database query
    mock_db.query.return_value.filter.return_value.all.return_value = mock_pois

    # Call the service function
    result = search_pois(mock_db, query)

    # Assertions
    assert result == mock_pois
    mock_db.query.return_value.filter.return_value.all.assert_called_once()