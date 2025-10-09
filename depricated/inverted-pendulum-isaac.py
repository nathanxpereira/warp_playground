from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import numpy as np
import torch
from abc import abstractmethod
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid, DynamicCylinder, FixedCuboid
from isaacsim.core.api.tasks import BaseTask
from isaacsim.core.cloner import GridCloner
from isaacsim.core.prims import RigidPrim, XFormPrim
import omni.usd
from pxr import Gf, UsdPhysics, UsdGeom

class InvertedPendulum(BaseTask):
    def __init__(self, name, num_envs, env_spacing, offset=None) -> None:
        BaseTask.__init__(self, name=name, offset=offset)
        self._num_envs = num_envs
        self._env_spacing = env_spacing
        self._cloner = GridCloner(self._env_spacing)
        
        # Pendulum parameters
        self.cart_length = 0.4
        self.cart_width = 0.3
        self.cart_height = 0.2
        self.pendulum_length = 1.0
        self.pendulum_radius = 0.02
        
        # Create track
        self.track_length = 12.0
        self.track_width = 0.3
        self.track_height = 0.05

        self.joint_size = 0.06


    def set_up_scene(self, scene) -> None:
        super().set_up_scene(scene)
        scene.add_default_ground_plane()
        self.set_object()
        
        # Clone the entire pendulum system for multi-env (skip if only 1 env)
        if self._num_envs > 1:
            prim_paths = self._cloner.generate_paths("/World/pendulum_system", self._num_envs - 1)
            # Generate paths starting from 1 to avoid conflict with original _0
            prim_paths = [f"/World/pendulum_system_{i+1}" for i in range(self._num_envs - 1)]
            self._cloner.clone(
                source_prim_path="/World/pendulum_system_0",
                prim_paths=prim_paths,
                position_offsets=np.array([[(i+1) * self._env_spacing, 0, 0] for i in range(self._num_envs - 1)]),
            )
            self._pendulum_system = XFormPrim(f"/World/pendulum_system_[0-{self._num_envs-1}]", "pendulum_system_view")
        else:
            self._pendulum_system = XFormPrim("/World/pendulum_system_0", "pendulum_system_view")
        scene.add(self._pendulum_system)

    def set_object(self):
        # Create parent XForm at track center height
        stage = omni.usd.get_context().get_stage()
        xform_prim = stage.DefinePrim("/World/pendulum_system_0", "Xform")
        UsdGeom.Xformable(xform_prim).AddTranslateOp().Set(Gf.Vec3d(0, 0, 0))
        
        # Create XFormPrim wrapper
        self.pendulum_system = XFormPrim("/World/pendulum_system_0", "pendulum_system_0")
        
        # Create all components under the parent (positions relative to parent at track center)
        self.track = FixedCuboid(
            prim_path="/World/pendulum_system_0/Track",
            name="track",
            position=np.array([0.0, 0.0, self.track_height/2]),
            size=1.0,
            scale=np.array([self.track_length, self.track_width, self.track_height]),
            color=np.array([0.5, 0.5, 0.5])
        )
        
        self.cart = DynamicCuboid(
            prim_path="/World/pendulum_system_0/Cart",
            name="cart",
            position=np.array([0.0, 0.0, self.track_height + self.cart_height/2]),
            size=1.0,
            scale=np.array([self.cart_length, self.cart_width, self.cart_height]),
            mass=None,
            density=1000.0,
            color=np.array([0.2, 0.5, 0.8])
        )
        
        self.pendulum = DynamicCylinder(
            prim_path="/World/pendulum_system_0/Pendulum", 
            name="pendulum",
            position=np.array([0.0, 0.0, self.track_height + self.cart_height + self.pendulum_length/2]),
            radius=self.pendulum_radius,
            height=self.pendulum_length,
            mass=None,
            density=1000.0,
            color=np.array([0.8, 0.2, 0.2])
        )



    def _create_joints(self):
        stage = omni.usd.get_context().get_stage()
        
        # Create joints before simulation starts
        for i in range(self._num_envs):
            env_path = f"/World/pendulum_system_{i}"
            
            # Prismatic joint: track top to cart bottom
            prismatic_prim = stage.DefinePrim(f"{env_path}/CartJoint", "PhysicsPrismaticJoint")
            prismatic = UsdPhysics.PrismaticJoint(prismatic_prim)
            prismatic.CreateBody0Rel().SetTargets([f"{env_path}/Track"])
            prismatic.CreateBody1Rel().SetTargets([f"{env_path}/Cart"])
            prismatic.CreateLocalPos0Attr(Gf.Vec3f(0, 0, self.track_height/2))  # Top of track (relative to track center)
            prismatic.CreateLocalPos1Attr(Gf.Vec3f(0, 0, -self.cart_height/2))  # Bottom of cart (relative to cart center)
            prismatic.CreateAxisAttr().Set("X")
            prismatic.CreateLowerLimitAttr().Set(-5.0)
            prismatic.CreateUpperLimitAttr().Set(5.0)

            # Revolute joint: cart top to pendulum origin
            joint_prim = stage.DefinePrim(f"{env_path}/PendulumJoint", "PhysicsRevoluteJoint")
            joint = UsdPhysics.RevoluteJoint(joint_prim)
            joint.CreateBody0Rel().SetTargets([f"{env_path}/Cart"])
            joint.CreateBody1Rel().SetTargets([f"{env_path}/Pendulum"])
            joint.CreateLocalPos0Attr(Gf.Vec3f(0, 0, self.cart_height/2))  # Top of cart
            joint.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))  # Pendulum origin (pivot point)
            joint.CreateAxisAttr().Set("Y")
        
    
        

    def get_observations(self) -> dict:
        object_positions, _ = self._pendulum_system.get_world_poses()
        return {self._pendulum_system.name: {"positions": object_positions}}

    def calculate_metrics(self) -> None:
        return torch.zeros(self._num_envs, device=self._device)

    def is_done(self) -> None:
        return torch.zeros(self._num_envs, device=self._device)

def main():
    num_envs = 1
    env_spacing = 3.0
    
    my_world = World(stage_units_in_meters=1.0, physics_prim_path="/physicsScene", backend="torch", device="cuda:0")
    my_task = InvertedPendulum(name="inverted_pendulum", num_envs=num_envs, env_spacing=env_spacing)
    my_world.add_task(my_task)
    my_world.reset()
    
    my_task._create_joints()
    
    # Save USD file
    stage = omni.usd.get_context().get_stage()
    stage.Export("inverted_pendulum.usd")
    print("Simulation saved to inverted_pendulum.usd")
    
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