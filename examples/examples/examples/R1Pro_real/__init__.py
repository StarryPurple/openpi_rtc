"""
R1 Pro Real Environment Package

This package provides a real-world environment interface for the Galaxea R1 Pro robot.
It is designed to work with the openpi_client runtime for policy execution.

Main components:
- constants: Robot-specific constants and parameters
- real_env: Core environment class for R1 Pro robot control
- env: OpenPI-compatible environment wrapper
- main: Main script for running with policy server

Usage:
    python -m r1pro_real.main --host 0.0.0.0 --port 8000

Example:
    from r1pro_real import env

    r1_env = env.make_r1pro_env(
        enable_cameras=True,
        render_height=224,
        render_width=224,
    )

    r1_env.reset()
    obs = r1_env.get_observation()
"""

from . import constants
from .real_env import RealEnv, make_real_env
from .env import R1ProRealEnvironment

__all__ = [
    "constants",
    "R1ProRealEnv",
    "make_real_env",
    "R1ProRealEnvironment",
    "make_r1pro_env",
]

__version__ = "0.1.0"
