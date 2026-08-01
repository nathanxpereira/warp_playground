"""CartPole simulation using Isaac Sim's classic cartpole asset.

This module loads the pre-built cartpole from Isaac Sim's asset library
and applies LQR control with state logging to CSV files.
"""

import numpy as np
import torch
import csv
from pathlib import Path

import os
from datetime import datetime

# Controllers
from config import CartPolePhysicsConfig
from controllers import LQRController

def get_cartpole_state(cartpole) -> dict:
    """Extract state from the cartpole articulation.

    Args:
        cartpole: Isaac Sim SingleArticulation instance (stable Core API)

    Returns:
        Dictionary with cart_position, cart_velocity, pole_angle, pole_angular_vel
    """
    joint_positions = cartpole.get_joint_positions()  # Shape: (2,) for single env
    joint_velocities = cartpole.get_joint_velocities()  # Shape: (2,) for single env

    # State vector: [cart_pos, cart_vel, pole_angle, pole_angular_vel]
    cart_position = float(joint_positions[0])  # Slider joint position
    cart_velocity = float(joint_velocities[0])
    pole_angle = float(joint_positions[1])  # Revolute joint angle
    pole_angular_vel = float(joint_velocities[1])

    return {
        "cart_position": cart_position,
        "cart_velocity": cart_velocity,
        "pole_angle": pole_angle,
        "pole_angular_vel": pole_angular_vel
    }


def apply_lqr_control(cartpole, controller, state: dict) -> float:
    """Apply LQR control to the cartpole.

    Args:
        cartpole: Isaac Sim SingleArticulation instance (stable Core API)
        controller: LQR controller
        state: Current state dictionary

    Returns:
        Control force applied
    """
    # Convert state to observations format expected by controller
    # Note: The controller expects quaternion format for pole_rotation
    # We create a simple quaternion from the pole angle (rotation around X-axis)
    angle = state["pole_angle"]
    qw = torch.cos(torch.tensor(angle) / 2).item()
    qx = torch.sin(torch.tensor(angle) / 2).item()

    observations = {
        "cart_position": torch.tensor([[state["cart_position"]]], device="cuda"),
        "cart_velocity": torch.tensor([[state["cart_velocity"]]], device="cuda"),
        "pole_rotation": torch.tensor([[qx, 0.0, 0.0, qw]], device="cuda"),  # Quaternion [x, y, z, w]
        "pole_velocity": torch.tensor([[0.0, 0.0, 0.0, 0.0, state["pole_angular_vel"], 0.0]], device="cuda")
    }

    # Compute control force
    control_force = controller.compute_control(observations, device="cuda")

    # Apply action to cart joint (index 0)
    # Force on cart (index 0), no torque on pole (index 1)
    action = ArticulationAction(
        joint_efforts=np.array([control_force.item(), 0.0])
    )
    cartpole.apply_action(action)

    return control_force.item()


def run_simulation(my_world, cartpole, controller, num_steps: int = 20):
    """Execute main simulation loop with LQR control and state logging.

    Args:
        my_world: Isaac Sim World instance
        cartpole: Cartpole articulation
        controller: LQR controller
        num_steps: Number of simulation steps to run
    """
    # Reset world
    print("Resetting world...", flush=True)
    my_world.reset()

    # Create log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path("cartpole/logs/")
    if not log_dir.exists(): log_dir.mkdir(parents=True)
    log_file = f"cartpole/logs/state_log_{timestamp}.csv"
    
    print(f"Starting simulation for {num_steps} steps...", flush=True)
    print(f"Logging state to: {log_file}", flush=True)

    # Debug flag for first few steps
    debug_steps = 3

    with open(log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["step", "cart_position", "cart_velocity", "pole_angle", "pole_angular_vel", "control_force"])

        # Simulation loop
        for step in range(num_steps):
            # Get state
            state = get_cartpole_state(cartpole)

            # Compute and apply control
            control_force = apply_lqr_control(cartpole, controller, state)

            # Debug output for first few steps
            if True:
                print(f"\n  DEBUG Step {step}:", flush=True)
                print(f"    State: pos={state['cart_position']:.4f}, vel={state['cart_velocity']:.4f}, "
                      f"angle={state['pole_angle']:.4f}, ang_vel={state['pole_angular_vel']:.4f}", flush=True)
                print(f"    Computed force: {control_force:.4f} N", flush=True)

            # Log state
            writer.writerow([
                step,
                state["cart_position"],
                state["cart_velocity"],
                state["pole_angle"],
                state["pole_angular_vel"],
                control_force
            ])

            # Step physics
            my_world.step(render=True)  # Enable rendering to see the simulation

            # Print progress with control force info
            if (step + 1) % 1 == 0:
                print(f"  Step {step + 1}/{num_steps} - Pole angle: {state['pole_angle']:.4f} rad, "
                      f"Cart pos: {state['cart_position']:.4f} m, Force: {control_force:.2f} N", flush=True)

    print(f"\nSimulation complete! State log saved to: {log_file}", flush=True)


if __name__ == "__main__":
    # Force NVIDIA GPU on dual-GPU laptops
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    print(">>> Loading SimulationApp (headless mode)...", flush=True)
    from isaacsim import SimulationApp

    CONFIG = {
        "headless": False,
        "active_gpu": 0,
        "physics_gpu": 0,
        "multi_gpu": False,
    }

    simulation_app = SimulationApp(CONFIG)
    
    from isaacsim.core.api import World
    from isaacsim.core.prims import SingleArticulation
    from isaacsim.core.utils.stage import add_reference_to_stage
    from isaacsim.storage.native import get_assets_root_path
    from isaacsim.core.utils.types import ArticulationAction

    my_world = None
    try:
        assets_root_path = get_assets_root_path()
        asset_path = assets_root_path + "/Isaac/Robots/IsaacSim/Cartpole/cartpole.usd"

        # Create world with standard units (1.0 meter)
        my_world = World(stage_units_in_meters=1.0, backend="torch", device="cuda")
        my_world.scene.add_default_ground_plane()

        # Add cartpole asset to stage
        add_reference_to_stage(usd_path=asset_path, prim_path="/World/cartpole")

        # Wrap as articulation for easy control
        cartpole = my_world.scene.add(
            SingleArticulation(prim_path="/World/cartpole", name="cartpole")
        )

        # Reset to initialize the scene
        my_world.reset()

        # Debug: Print joint information
        print(f"\n=== Joint Information ===", flush=True)
        print(f"Number of DOF: {cartpole.num_dof}", flush=True)
        print(f"Joint names: {cartpole.dof_names}", flush=True)

        # Extract physical properties from the USD asset
        from pxr import UsdPhysics, Usd, UsdGeom
        stage = my_world.stage

        print(f"\n=== USD Asset Properties ===", flush=True)

        # Initialize variables for extracted properties
        cart_mass = None
        pole_mass = None
        pole_length = None

        # Extract properties directly from known paths (based on USD structure)
        # Cart rigid body
        cart_prim = stage.GetPrimAtPath("/World/cartpole/cart")
        pole_prim = stage.GetPrimAtPath("/World/cartpole/pole")
        pole_shape_prim = pole_prim.GetPrimAtPath("visuals/mesh_0")

        cart_mass = cart_prim.GetAttribute('physics:mass').Get()
        pole_mass = pole_prim.GetAttribute('physics:mass').Get()
        pole_size = np.array(pole_shape_prim.GetAttribute('xformOp:scale').Get())
        
        # Configure joint drives for force control at USD level
        # Disable stiffness/damping on cart joint (slider) to allow pure force control
        cart_joint = UsdPhysics.DriveAPI.Get(stage.GetPrimAtPath("/World/cartpole/slider_to_cart"), "linear")
        if cart_joint:
            cart_joint.GetStiffnessAttr().Set(0.0)
            cart_joint.GetDampingAttr().Set(0.0)
            print(f"Configured cart joint for force control (stiffness=0, damping=0)", flush=True)

        # Pole joint should be free to rotate (no control)
        pole_joint = UsdPhysics.DriveAPI.Get(stage.GetPrimAtPath("/World/cartpole/cart_to_pole"), "angular")
        if pole_joint:
            pole_joint.GetStiffnessAttr().Set(0.0)
            pole_joint.GetDampingAttr().Set(0.0)
            print(f"Configured pole joint as free joint (stiffness=0, damping=0)", flush=True)

        print(f"=========================\n", flush=True)

        # Create config with default values, then override with USD values
        physics_config = CartPolePhysicsConfig(cart_mass=cart_mass, pole_mass=pole_mass, pole_length=pole_length)

        A, B = physics_config.compute_system_matrices()

        # Define LQR cost matrices
        Q = np.diag([10.0, 1.0, 1.0, 1.0])  # State cost weights [cart_pos, cart_vel, pole_angle, pole_vel]
        R = np.array([[0.1]])  # Control cost weight

        controller = LQRController(A, B, Q, R)

        # Run the simulation
        run_simulation(my_world, cartpole, controller, num_steps=50)

        # Keep IsaacSim open for user interaction
        print("\n" + "="*60, flush=True)
        print("Simulation steps complete!", flush=True)
        print("IsaacSim will remain open for inspection.", flush=True)
        print("Close the window when you're done.", flush=True)
        print("="*60 + "\n", flush=True)

        # Keep the app running until user closes it manually
        while simulation_app.is_running():
            simulation_app.update()

    except Exception as e:
        print(f"\n*** ERROR: {e} ***", flush=True)
        import traceback
        traceback.print_exc()
    finally:
        print("\nShutting down...", flush=True)
        # Windows workaround: add multiple update() calls before close()
        # to properly clean up threads (known issue in Isaac Sim on Windows)
        for _ in range(10):
            simulation_app.update()
        simulation_app.close()
