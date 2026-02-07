"""CartPole simulation environment for Isaac Sim.

This module implements a cart-pole (inverted pendulum) system using
NVIDIA Isaac Sim's physics engine. The system consists of a cart
constrained to move horizontally and a pole connected via a revolute
joint.

The implementation supports:
- Multiple parallel environments (num_envs parameter)
- Tensor-based state observations for efficient computation
- Integration with various controller types (LQR, PID, etc.)
- Configurable physical parameters via CartPolePhysicsConfig
"""


import numpy as np
import torch

from isaacsim import SimulationApp
from isaacsim.core.api import World
from isaacsim.core.api.tasks import BaseTask
from isaacsim.core.prims import RigidPrim
import omni.usd
from pxr import Gf, UsdPhysics, UsdGeom, Sdf

from cartpole.controllers import LQRController
from config import CartPolePhysicsConfig, SimulationConfig

simulation_app = SimulationApp({"headless": False})


class CartPole(BaseTask):
    """Cartpole task for Isaac Sim.

    Implements an inverted pendulum system with a cart constrained to
    horizontal motion and a pole that can rotate freely about the cart.
    """
    def __init__(self, name: str, num_envs: int, config: CartPolePhysicsConfig = None, offset=None) -> None:
        """Initialize CartPole task.

        Args:
            name: Name of the task
            num_envs: Number of parallel environments
            config: Physical configuration parameters (uses defaults if None)
            offset: Offset for environment placement
        """
        BaseTask.__init__(self, name=name, offset=offset)
        self._num_envs = num_envs
        self.config = config or CartPolePhysicsConfig()

        # Compute system matrices from config
        self.A, self.B = self.config.compute_system_matrices()

    def set_up_scene(self, scene) -> None:
        """Set up the simulation scene with ground plane and cartpole.

        Args:
            scene: Isaac Sim scene to add objects to
        """
        super().set_up_scene(scene)
        scene.add_default_ground_plane()
        self._create_cartpole()
        
        self._carts = RigidPrim("/World/Cart_0", "cart")
        self._poles = RigidPrim("/World/Pole_0", "pole")
        scene.add(self._carts)
        scene.add(self._poles)

    def _create_cart_geometry(self, stage, cart_path: str) -> None:
        """Create cart geometry with collision and physics properties."""
        cart_prim = stage.DefinePrim(cart_path, "Xform")
        cart_geom = UsdGeom.Cube.Define(stage, f"{cart_path}/Geom")
        cart_geom.CreateSizeAttr(1.0)

        # Scale and position
        cart_geom.GetPrim().CreateAttribute("xformOp:scale", Sdf.ValueTypeNames.Float3).Set(
            Gf.Vec3f(self.config.cart_length, self.config.cart_width, self.config.cart_height)
        )
        cart_geom.GetPrim().CreateAttribute("xformOp:translate", Sdf.ValueTypeNames.Double3).Set(
            Gf.Vec3d(0, 0, self.config.cart_height/2)
        )
        cart_geom.CreateXformOpOrderAttr().Set(["xformOp:translate", "xformOp:scale"])

        # Physics
        UsdPhysics.CollisionAPI.Apply(cart_geom.GetPrim())
        UsdPhysics.RigidBodyAPI.Apply(cart_prim)
        UsdPhysics.MassAPI.Apply(cart_prim).CreateMassAttr(self.config.cart_mass)

    def _create_pole_geometry(self, stage, pole_path: str) -> None:
        """Create pole geometry with initial angle and physics properties."""
        pole_prim = stage.DefinePrim(pole_path, "Xform")
        UsdGeom.Xformable(pole_prim).AddTranslateOp().Set(
            Gf.Vec3d(0, 0, self.config.cart_height)
        )
        UsdGeom.Xformable(pole_prim).AddRotateYOp().Set(self.config.start_angle_degrees)

        pole_geom = UsdGeom.Cylinder.Define(stage, f"{pole_path}/Geom")
        pole_geom.CreateRadiusAttr(self.config.pole_radius)
        pole_geom.CreateHeightAttr(self.config.pole_length)
        pole_geom.GetPrim().CreateAttribute("xformOp:translate", Sdf.ValueTypeNames.Double3).Set(
            Gf.Vec3d(0, 0, self.config.pole_length/2)
        )
        pole_geom.CreateXformOpOrderAttr().Set(["xformOp:translate"])

        # Physics
        UsdPhysics.CollisionAPI.Apply(pole_geom.GetPrim())
        UsdPhysics.RigidBodyAPI.Apply(pole_prim)
        UsdPhysics.MassAPI.Apply(pole_prim).CreateMassAttr(self.config.pole_mass)

    def _create_cart_joint(self, stage) -> None:
        """Create prismatic joint constraining cart to X-axis."""
        cart_joint = stage.DefinePrim("/World/CartJoint", "PhysicsPrismaticJoint")
        joint = UsdPhysics.PrismaticJoint(cart_joint)
        joint.CreateBody1Rel().SetTargets(["/World/Cart_0"])
        joint.CreateAxisAttr("X")
        joint.CreateLowerLimitAttr(-5.0)
        joint.CreateUpperLimitAttr(5.0)

    def _create_pole_joint(self, stage) -> None:
        """Create revolute joint connecting pole to cart."""
        pole_joint = stage.DefinePrim("/World/PoleJoint", "PhysicsRevoluteJoint")
        joint = UsdPhysics.RevoluteJoint(pole_joint)
        joint.CreateBody0Rel().SetTargets(["/World/Cart_0"])
        joint.CreateBody1Rel().SetTargets(["/World/Pole_0"])
        joint.CreateLocalPos0Attr(Gf.Vec3f(0, 0, self.config.cart_height))  # Top of cart
        joint.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))  # Bottom of pole (at pole origin)
        joint.CreateAxisAttr("Y")

    def _create_cartpole(self) -> None:
        """Create complete cartpole USD scene with geometry and joints."""
        stage = omni.usd.get_context().get_stage()

        self._create_cart_geometry(stage, "/World/Cart_0")
        self._create_pole_geometry(stage, "/World/Pole_0")
        self._create_cart_joint(stage)
        self._create_pole_joint(stage)

    def get_observations(self) -> dict:
        """Get current state observations for all environments.

        Returns:
            Dictionary with keys:
                - cart_position: (num_envs, 3) tensor of cart positions
                - cart_velocity: (num_envs, 6) tensor of cart velocities
                - pole_rotation: (num_envs, 4) tensor of pole quaternions
                - pole_velocity: (num_envs, 6) tensor of pole velocities
        """
        cart_pos, _ = self._carts.get_world_poses()
        _, pole_rot = self._poles.get_world_poses()
        cart_vel = self._carts.get_velocities()
        pole_vel = self._poles.get_velocities()
        return {
            "cart_position": cart_pos,
            "pole_rotation": pole_rot,
            "cart_velocity": cart_vel,
            "pole_velocity": pole_vel
        }
    
    def apply_control(self, force: torch.Tensor) -> None:
        """Apply horizontal control force to cart(s).

        Args:
            force: Tensor of shape (num_envs,) with X-axis forces in Newtons
        """
        forces = torch.zeros((self._num_envs, 3), device=self._device)
        forces[:, 0] = force  # X-axis force
        self._carts.apply_forces(forces)

    def calculate_metrics(self) -> torch.Tensor:
        """Calculate performance metrics for each environment.

        TODO: Implement metrics calculation (e.g., distance from upright,
              cart position bounds violation, energy consumption)

        Returns:
            Tensor of shape (num_envs,) with metric values
        """
        return torch.zeros(self._num_envs, device=self._device)

    def is_done(self) -> torch.Tensor:
        """Check if episode termination conditions are met.

        TODO: Implement termination logic (e.g., pole angle > threshold,
              cart position out of bounds, time limit exceeded)

        Returns:
            Boolean tensor of shape (num_envs,) indicating episode completion
        """
        return torch.zeros(self._num_envs, dtype=torch.bool, device=self._device)

    def reset_state(self) -> None:
        """Reset all environments to initial state with configured start angle."""
        angle_rad = np.radians(self.config.start_angle_degrees)
        pole_quat = torch.tensor([[np.cos(angle_rad/2), 0, np.sin(angle_rad/2), 0]], dtype=torch.float32, device=self._device)
        self._carts.set_world_poses(torch.tensor([[0, 0, self.config.cart_height/2]], dtype=torch.float32, device=self._device))
        self._poles.set_world_poses(torch.tensor([[0, 0, self.config.cart_height]], dtype=torch.float32, device=self._device), pole_quat)
        self._carts.set_velocities(torch.zeros((1, 6), dtype=torch.float32, device=self._device))
        self._poles.set_velocities(torch.zeros((1, 6), dtype=torch.float32, device=self._device))



def _export_scene_to_usd(filepath: str) -> None:
    """Export current scene to USD file.

    Args:
        filepath: Path where USD file should be saved
    """
    stage = omni.usd.get_context().get_stage()
    stage.Export(filepath)
    print(f"Simulation scene saved to {filepath}")


def _create_lqr_controller(task: CartPole, Q, R) -> LQRController:
    """Create LQR controller with default cost matrices.

    Args:
        task: CartPole task with system matrices A and B

    Returns:
        Configured LQR controller
    """
    return LQRController(task.A, task.B, Q, R)


def _run_simulation(world: World, task: CartPole, controller, config: SimulationConfig) -> None:
    """Execute main simulation loop with reset logic.

    Args:
        world: Isaac Sim world
        task: CartPole task
        controller: Control policy (LQR, PID, etc.)
        config: Simulation configuration parameters
    """
    reset_needed = True

    try:
        while simulation_app.is_running():
            if world.is_stopped() and not reset_needed:
                reset_needed = True

            if world.is_playing():
                if reset_needed:
                    world.reset(soft=True)
                    task.reset_state()

                    # Warmup physics simulation
                    for _ in range(config.warmup_steps):
                        world.step(render=False)

                    controller.reset()
                    reset_needed = False

                # Control loop
                observations = world.get_observations()
                force = controller.compute_control(observations, world.device)
                task.apply_control(force)

            world.step(render=config.render)
    finally:
        world.stop()


def main():
    """Main entry point for cartpole simulation."""
    # Configuration
    sim_config = SimulationConfig(warmup_steps=3, render=True)
    physics_config = CartPolePhysicsConfig()

    # World setup
    my_world = World(
        stage_units_in_meters=1.0,
        physics_prim_path="/physicsScene",
        backend="torch",
        device="cuda:0"
    )

    # Task creation
    my_task = CartPole(name="cartpole", num_envs=1, config=physics_config)
    my_world.add_task(my_task)
    my_world.reset()
    my_world.pause()

    # Optional: Export USD for visualization
    _export_scene_to_usd("cartpole/cartpole.usd")

    # Controller setup
    Q = np.diag([1.0, 1.0, 1.0, 1.0])
    R = np.array([[1.0]])
    controller = _create_lqr_controller(my_task, Q, R)

    # Main simulation loop
    _run_simulation(my_world, my_task, controller, sim_config)

    # Cleanup
    simulation_app.close()


if __name__ == "__main__":
    main()
