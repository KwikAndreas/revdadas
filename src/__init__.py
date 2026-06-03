"""
RevDadas - AI-Driven Predictive Analytics for Fraud Detection and Revenue Forecasting
"""

__version__ = "0.1.0"
__author__ = "RevDadas Team"

from . import data_loader
from . import preprocessing
from . import forecasting
from . import anomaly_detection
from . import utils
from . import policy
from . import report
from . import business

__all__ = [
    "data_loader",
    "preprocessing",
    "forecasting",
    "anomaly_detection",
    "utils",
    "policy",
    "report",
]
