import os
import pytest


@pytest.fixture(scope="session")
def apsimx_dir():
    r"""str: Path to the apsimx directory."""
    return os.path.dirname(
        os.path.dirname(os.path.dirname(__file__)))


@pytest.fixture(scope="session")
def example_model(apsimx_dir):
    r"""str: Path to example model input file."""
    return os.path.join(apsimx_dir, "Examples", "Wheat.apsimx")
