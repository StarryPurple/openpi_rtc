"""
Main script for running R1 Pro robot with policy server.

This script connects to a policy server via websocket and executes
actions on the real R1 Pro robot.

Usage:
    python -m r1pro_real.main --host 0.0.0.0 --port 8000 --action_horizon 25
"""

import dataclasses
import logging
import numpy as np
from typing import Optional, Dict

from openpi_client import action_chunk_broker
from openpi_client import websocket_client_policy as _websocket_client_policy
from openpi_client.runtime import runtime as _runtime
from openpi_client.runtime.agents import policy_agent as _policy_agent
import tyro

from R1Pro_real import env as _env
from R1Pro_real import constants


@dataclasses.dataclass
class Args:
    """Command line arguments for R1 Pro main script."""

    # Server configuration
    host: str = "0.0.0.0"
    """Host address of the policy server"""

    port: int = 8000
    """Port of the policy server"""

    # Action configuration
    action_horizon: int = 25
    """Number of action steps to execute per policy query"""

    # Episode configuration
    num_episodes: int = 1
    """Number of episodes to run"""

    max_episode_steps: int = 1000
    """Maximum number of steps per episode"""

    # Runtime configuration
    max_hz: float = 50.0
    """Maximum control frequency (Hz)"""

    # Camera configuration
    enable_cameras: bool = True
    """Whether to enable camera observations"""

    render_height: int = 224
    """Height to resize camera images to"""

    render_width: int = 224
    """Width to resize camera images to"""

    # Robot configuration
    enable_torso: bool = False
    """Whether to enable torso control"""

    enable_mobile_base: bool = False
    """Whether to enable mobile base control"""

    # Reset configuration
    reset_left_arm: Optional[str] = None
    """Comma-separated values for left arm reset position (7 values)"""

    reset_right_arm: Optional[str] = None
    """Comma-separated values for right arm reset position (7 values)"""

    reset_torso: Optional[str] = None
    """Comma-separated values for torso reset position (4 values)"""


def parse_reset_position(args: Args) -> Optional[Dict[str, np.ndarray]]:
    """
    Parse reset position from command line arguments.

    Args:
        args: Command line arguments

    Returns:
        Dictionary with reset positions or None
    """
    reset_position = {}

    if args.reset_left_arm is not None:
        try:
            values = [float(x.strip()) for x in args.reset_left_arm.split(",")]
            if len(values) != 7:
                raise ValueError(f"Expected 7 values for left arm, got {len(values)}")
            reset_position["left_arm"] = np.array(values)
        except Exception as e:
            logging.error(f"Failed to parse left arm reset position: {e}")
            return None

    if args.reset_right_arm is not None:
        try:
            values = [float(x.strip()) for x in args.reset_right_arm.split(",")]
            if len(values) != 7:
                raise ValueError(f"Expected 7 values for right arm, got {len(values)}")
            reset_position["right_arm"] = np.array(values)
        except Exception as e:
            logging.error(f"Failed to parse right arm reset position: {e}")
            return None

    if args.reset_torso is not None:
        try:
            values = [float(x.strip()) for x in args.reset_torso.split(",")]
            if len(values) != 4:
                raise ValueError(f"Expected 4 values for torso, got {len(values)}")
            reset_position["torso"] = np.array(values)
        except Exception as e:
            logging.error(f"Failed to parse torso reset position: {e}")
            return None

    return reset_position if reset_position else None


def main(args: Args) -> None:
    """
    Main function to run R1 Pro robot with policy server.

    Args:
        args: Command line arguments
    """
    # Connect to policy server
    logging.info(f"Connecting to policy server at {args.host}:{args.port}")
    ws_client_policy = _websocket_client_policy.WebsocketClientPolicy(
        host=args.host,
        port=args.port,
    )

    # Get server metadata
    metadata = ws_client_policy.get_server_metadata()
    logging.info(f"Server metadata: {metadata}")

    # Parse reset position
    reset_position = parse_reset_position(args)

    # Override with server metadata if available
    if "reset_pose" in metadata:
        reset_position = metadata.get("reset_pose")
        logging.info(f"Using reset pose from server metadata")

    if reset_position is not None:
        logging.info(f"Reset position:")
        for key, value in reset_position.items():
            logging.info(f"  {key}: {value}")
    else:
        logging.info(f"Using default reset position from constants.py")

    # Create environment
    logging.info("Creating R1 Pro environment")
    environment = _env.R1ProRealEnvironment(
        reset_position=reset_position,
        render_height=args.render_height,
        render_width=args.render_width,
        enable_cameras=args.enable_cameras,
        enable_torso=args.enable_torso,
        enable_mobile_base=args.enable_mobile_base,
    )

    # Create policy agent with action chunking
    policy = action_chunk_broker.ActionChunkBroker(
        policy=ws_client_policy,
        action_horizon=args.action_horizon,
    )

    agent = _policy_agent.PolicyAgent(policy=policy)

    # Create runtime
    logging.info("Creating runtime")
    runtime = _runtime.Runtime(
        environment=environment,
        agent=agent,
        subscribers=[],
        max_hz=args.max_hz,
        num_episodes=args.num_episodes,
        max_episode_steps=args.max_episode_steps,
    )

    # Run episodes
    logging.info(f"Starting {args.num_episodes} episode(s)")
    try:
        runtime.run()
    except KeyboardInterrupt:
        logging.info("Interrupted by user")
    finally:
        logging.info("Closing environment")
        environment.close()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True
    )
    tyro.cli(main)
