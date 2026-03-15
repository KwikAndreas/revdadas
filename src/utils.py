"""
Utility functions for RevDadas project
"""

import os
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_project_root():
    """Get root directory of the project"""
    return Path(__file__).parent.parent


def get_data_path(subfolder="raw"):
    """Get path to data directory"""
    root = get_project_root()
    path = root / "data" / subfolder
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_models_path():
    """Get path to models directory"""
    root = get_project_root()
    path = root / "models"
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_directory(path):
    """Ensure directory exists"""
    os.makedirs(path, exist_ok=True)
    return path


def format_currency(value, short=False):
    """Format value to currency format (Rp)
    
    Args:
        value: numeric value
        short: if True, format as Rp XXX T (trillion), M (million), etc.
    """
    if not isinstance(value, (int, float)):
        return value
    
    if short:
        if abs(value) >= 1e12:
            return f"Rp {value/1e12:.1f} T"
        elif abs(value) >= 1e9:
            return f"Rp {value/1e9:.1f} M"
        elif abs(value) >= 1e6:
            return f"Rp {value/1e6:.1f} K"
    
    return f"Rp {value:,.0f}"


def format_currency_detailed(value):
    """Format currency with full decimal for display"""
    if isinstance(value, (int, float)):
        return f"Rp {value:,.2f}"
    return value


def format_percentage(value, decimals=1):
    """Format value to percentage"""
    if isinstance(value, (int, float)):
        return f"{value:.{decimals}f}%"
    return value
