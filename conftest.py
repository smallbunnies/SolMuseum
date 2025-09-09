import pytest

@pytest.fixture
def rtol():
    return 1e-5

@pytest.fixture
def atol():
    return 1e-6
