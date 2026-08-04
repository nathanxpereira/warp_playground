# logistics
import os
from pathlib import Path
from datetime import datetime

# compute libraries
import numpy as np
import polars as pl
import torch

# Controllers
from config import CartPolePhysicsConfig
from controllers import LQRController

from isaacsim import SimulationApp

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
CONFIG = {
    "headless": False,
    "active_gpu": 0,
    "physics_gpu": 0,
    "multi_gpu": False,
}
simulation_app = SimulationApp(CONFIG)

# import world
from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.storage.native import get_assets_root_path
from isaacsim.core.utils.types import ArticulationAction

from pxr import PhysxSchema, Usd


def get_cartpole_state(cartpole) -> dict:
    '''
    Extract state from cartpole USD object. 
    Args:
        cartpole USD object
    Returns:
        state dict: cart position, cart velocity, pole angle, and pole angular velocity

    TODO: Scale for multi environment example. 
    '''
    joint_positions = cartpole.get_joint_positions()
    joint_velocities = cartpole.get_joint_velocities()

    # State vector: [cart_pos, cart_vel, pole_angle, pole_angular_vel]
    cart_position = float(joint_positions[0])
    cart_velocity = float(joint_velocities[0])
    pole_angle = float(joint_positions[1])
    pole_angular_vel = float(joint_velocities[1])

    return {
        "cart_position": cart_position,
        "cart_velocity": cart_velocity,
        "pole_angle": pole_angle,
        "pole_angular_vel": pole_angular_vel
    }


def apply_control(cartpole, controller, state: dict) -> float:
    """
    Apply LQR control to the cartpole.
    Args:
        cartpole USD object, controller, and state
    Returns:
        Control force applied
    """
    observations = {
        "cart_position": torch.tensor([state["cart_position"]], device="cuda"),
        "cart_velocity": torch.tensor([state["cart_velocity"]], device="cuda"),
        "pole_rotation": torch.tensor([state["pole_angle"]], device="cuda"),
        "pole_velocity": torch.tensor([state["pole_angular_vel"]], device="cuda"),
    }

    control_force = controller.compute_control(observations, device="cuda")

    # joints = ['slider_to_cart', 'cart_to_pole'] 
    # apply force to the first joint, but not second
    joint_efforts = np.array([control_force.item(), 0.0])
    action = ArticulationAction(joint_efforts=joint_efforts)
    cartpole.apply_action(action)

    return control_force.item()

def set_initial_pole_state(stage, pole_angle_deg):
    pole_joint = stage.GetPrimAtPath("/World/cartpole/cart/cart_to_pole")
    state_api = PhysxSchema.JointStateAPI.Apply(pole_joint, "angular")
    state_api.CreatePositionAttr(float(pole_angle_deg))
    state_api.CreateVelocityAttr(0.0)


def build_world(initial_pole_angle_deg: float = 0.0):
    # build world and place cartpole
    assets_root_path = get_assets_root_path()
    asset_path = assets_root_path + "/Isaac/Robots/IsaacSim/Cartpole/cartpole.usd"

    dt = 1.0/60.0
    my_world = World(stage_units_in_meters=1.0, physics_dt=dt, backend="torch", device="cuda")
    my_world.scene.add_default_ground_plane(z_position=-0.1)
    add_reference_to_stage(usd_path=asset_path, prim_path="/World/cartpole")
    cartpole = my_world.scene.add(
        SingleArticulation(prim_path="/World/cartpole", name="cartpole")
    )

    set_initial_pole_state(my_world.stage, initial_pole_angle_deg)

    my_world.reset()

    # must be after reset to set the defaults. slider_to_cart not changed. 
    default_positions = torch.tensor([0.0, np.deg2rad(initial_pole_angle_deg)], dtype=torch.float32)
    cartpole.set_joints_default_state(positions=default_positions, velocities=torch.zeros(2, dtype=torch.float32))

    return my_world, cartpole

def set_physics(my_world):
    # Extract cartpole properties
    stage = my_world.stage
    cart_prim = stage.GetPrimAtPath("/World/cartpole/cart")
    pole_prim = stage.GetPrimAtPath("/World/cartpole/pole")
    pole_shape_prim = pole_prim.GetPrimAtPath("visuals/mesh_0")

    cart_mass = cart_prim.GetAttribute('physics:mass').Get()
    pole_mass = pole_prim.GetAttribute('physics:mass').Get()
    pole_size = np.array(pole_shape_prim.GetAttribute('xformOp:scale').Get())

    physics_config = CartPolePhysicsConfig(cart_mass=cart_mass, pole_mass=pole_mass, pole_length=pole_size[2])

    return physics_config


def run_simulation(my_world, cartpole, controller, duration: float = 1.0):
    """
    Main simulation loop with LQR control and state logging.

    Args:
        my_world, controller, duration (s)
    """

    my_world.reset()
    
    physics_dt = my_world.get_physics_dt()
    num_steps = round(duration / physics_dt)
    print(f"physics_dt={physics_dt:.6f}s  duration={duration}s  num_steps={num_steps}", flush=True)

    history = {
        "step": np.zeros(num_steps), 
        "time": np.zeros(num_steps),
        "cart_position": np.zeros(num_steps), 
        "cart_velocity": np.zeros(num_steps), 
        "pole_angle": np.zeros(num_steps), 
        "pole_angular_vel": np.zeros(num_steps), 
        "control_force": np.zeros(num_steps),
    }
    
    # Simulation loop
    for step in range(num_steps):
        # gather state and apply force
        state = get_cartpole_state(cartpole)
        control_force = apply_control(cartpole, controller, state)

        # Step physics
        my_world.step()  # Enable rendering to see the simulation

        t = my_world.current_time
        # Log state
        history['step'][step] = step
        history['time'][step] = t
        history['cart_position'][step] = state["cart_position"]
        history['cart_velocity'][step] = state["cart_velocity"]
        history['pole_angle'][step] = state["pole_angle"]
        history['pole_angular_vel'][step] = state["pole_angular_vel"]
        history['control_force'][step] = control_force

        if (step + 1) % 1 == 0:
            print(f"  Step {step + 1}/{num_steps} - t: {t:.4f} s, "
                    f"angle: {state['pole_angle']:.4f} rad, "
                    f"ang_vel: {state['pole_angular_vel']:.4f} rad/s, "
                    f"cart: {state['cart_position']:.4f} m, "
                    f"cart_vel: {state['cart_velocity']:.4f} m/s, Force: {control_force:.2f} N, ",
                    flush=True)

    df = pl.DataFrame(history)

    # Create log dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(f"cartpole/logs/log_{timestamp}")
    if not log_dir.exists(): log_dir.mkdir(parents=True)

    file = str(log_dir / "log.parquet")
    df.write_parquet(file)
    print(f"File written to: {file}")

def main(Q, R, pole_angle=0.0):
    my_world, cartpole = build_world(initial_pole_angle_deg=pole_angle)
    physics_config = set_physics(my_world)

    A, B = physics_config.compute_system_matrices()

    controller = LQRController(A, B, Q, R)

    run_simulation(my_world, cartpole, controller, duration=1)

    # Keep the app running. Does not restart sim. Need to fix. 
    while simulation_app.is_running():
        simulation_app.update()

    for _ in range(10):
        simulation_app.update()
    simulation_app.close()

if __name__ == "__main__":
    # Define LQR cost matrices
    pole_angle = 2.0
    Q = np.diag([1.0, 1.0, 10.0, 10.0])  # State cost weights [cart_pos, cart_vel, pole_angle, pole_vel]
    R = np.array([[0.1]])  # Control cost weight
    main(Q, R, pole_angle)    

    