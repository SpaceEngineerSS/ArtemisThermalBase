"""Pytest configuration and shared fixtures for ArtemisThermalBase tests.

Author: Mehmet Gümüş (github.com/SpaceEngineerSS)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest


# Add project root to path so imports work
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def pytest_configure(config: pytest.Config) -> None:
    """Configure logging for tests."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(name)s [%(levelname)s] %(message)s",
    )
