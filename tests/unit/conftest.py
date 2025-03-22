import pytest
from unittest.mock import MagicMock

@pytest.fixture
def mock_db():
    """Fixture to provide a mocked database session."""
    db = MagicMock()
    return db