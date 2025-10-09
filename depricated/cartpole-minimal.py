from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import numpy as np
import torch
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid, DynamicCylinder, FixedCuboid
from isaacsim.core.api.tasks import BaseTask
from isaacsim.core.cloner import GridCloner
from isaacsim.core.prims import RigidPrim
import omni.usd

class CartPole(BaseTask):
    def __init__(self, name, num_envs, env_spacing, offset=None) -> None:
        BaseTask.__init__(self, name=name, offset=offset)
        self._num_envs = num_envs
        self._env_spacing = env_spacing
        self._cloner = GridCloner(self._env_spacing)

    def set_up_scene(self, scene) -> None:
        super().set_up_scene(scene)
        scene.add_default_ground_plane()
        self.set_object()
        
        # Add objects to scene
        scene.add(self.track)
        scene.add(self.cart)
        scene.add(self.pendulum)
        
        # Clone for multi-env
        prim_paths = self._cloner.generate_paths("/World/pendulum", self._num_envs)
        self._cloner.clone(
            source_prim_path=self.pendulum.prim_path,
            prim_paths=prim_paths,
            position_offsets=np.array([[i * self._env_spacing, 0, 0] for i in range(self._num_envs)]),
        )
        self._pendulum = RigidPrim(prim_paths_expr=f"/World/pendulum_[0-{self._num_envs-1}]", name="pendulum_view")
        scene.add(self._pendulum)

    def set_object(self):
        # Track
        self.track = FixedCuboid(
            prim_path="/World/Track_0",
            name="track_0",
            position=np.array([0.0, 0.0, 0.025]),
            size=1.0,
            scale=np.array([12.0, 0.3, 0.05]),
            color=np.array([0.5, 0.5, 0.5])
        )
        
        # Cart - positioned on track
        self.cart = DynamicCuboid(
            prim_path="/World/Cart_0",
            name="cart_0",
            position=np.array([0.0, 0.0, 0.15]),
            size=1.0,
            scale=np.array([0.4, 0.3, 0.2]),
            mass=1.0,
            color=np.array([0.2, 0.5, 0.8])
        )
        
        # Pendulum - positioned above cart
        self.pendulum = DynamicCylinder(
            prim_path="/World/pendulum_0", 
            name="pendulum_0",
            position=np.array([0.0, 0.0, 0.65]),
            radius=0.02,
            height=1.0,
            mass=0.1,
            color=np.array([0.8, 0.2, 0.2])
        )

    def get_observations(self) -> dict:
        positions, _ = self._pendulum.get_world_poses()
        velocities = self._pendulum.get_velocities()
        return {self._pendulum.name: {"positions": positions, "velocities": velocities}}

    def calculate_metrics(self) -> None:
        return torch.zeros(self._num_envs, device=self._device)

    def is_done(self) -> None:
        return torch.zeros(self._num_envs, device=self._device)

def main():
    num_envs = 1
    env_spacing = 3.0
    
    my_world = World(stage_units_in_meters=1.0, physics_prim_path="/physicsScene", backend="torch", device="cuda:0")
    my_task = CartPole(name="cartpole", num_envs=num_envs, env_spacing=env_spacing)
    my_world.add_task(my_task)
    my_world.reset()
    
    # Save USD file
    stage = omni.usd.get_context().get_stage()
    stage.Export("cartpole_minimal.usd")
    print("Simulation saved to cartpole_minimal.usd")
    
    reset_needed = False
    try:
        while simulation_app.is_running():
            if my_world.is_stopped() and not reset_needed:
                reset_needed = True
            if my_world.is_playing():
                if reset_needed:
                    my_world.reset(soft=True)
                    reset_needed = False
                observations = my_world.get_observations()
            my_world.step(render=True)
    finally:
        my_world.stop()
        simulation_app.close()

if __name__ == "__main__":
    main()