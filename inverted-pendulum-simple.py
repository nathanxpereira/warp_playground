from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import numpy as np
import torch
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid, DynamicCylinder, FixedCuboid
from isaacsim.core.api.tasks import BaseTask
from isaacsim.core.cloner import GridCloner
from isaacsim.core.prims import RigidPrim, XFormPrim

import omni.usd
from pxr import Gf, UsdPhysics, PhysxSchema, UsdGeom, Sdf

class InvertedPendulum(BaseTask):
    def __init__(self, name, num_envs, env_spacing, offset=None) -> None:
        BaseTask.__init__(self, name=name, offset=offset)
        self._num_envs = num_envs
        self._env_spacing = env_spacing
        self._cloner = GridCloner(self._env_spacing)

    def set_up_scene(self, scene) -> None:
        super().set_up_scene(scene)
        scene.add_default_ground_plane()
        self._create_articulation()
        
        # Clone the pendulum
        prim_paths = self._cloner.generate_paths("/World/Pendulum", self._num_envs)
        self._cloner.clone(
            source_prim_path="/World/Pendulum_0",
            prim_paths=prim_paths,
            position_offsets=np.array([[i * self._env_spacing, 0, 0] for i in range(self._num_envs)]),
        )
        
        self._pendulums = RigidPrim(prim_paths_expr="/World/Pendulum_[0-9]*", name="pendulums")
        scene.add(self._pendulums)

    def _create_articulation(self):
        stage = omni.usd.get_context().get_stage()
        
        # Create track (fixed base)
        track_prim = stage.DefinePrim("/World/Track_0", "Xform")
        track_geom = UsdGeom.Cube.Define(stage, "/World/Track_0/Geometry")
        track_geom.CreateSizeAttr(1.0)
        track_geom.GetPrim().CreateAttribute("xformOp:scale", Sdf.ValueTypeNames.Float3).Set(Gf.Vec3f(12.0, 0.3, 0.05))
        track_geom.GetPrim().CreateAttribute("xformOp:translate", Sdf.ValueTypeNames.Double3).Set(Gf.Vec3d(0, 0, 0.025))
        track_geom.CreateXformOpOrderAttr().Set(["xformOp:translate", "xformOp:scale"])
        
        # Add collision and rigid body to track
        UsdPhysics.CollisionAPI.Apply(track_geom.GetPrim())
        rigid_api = UsdPhysics.RigidBodyAPI.Apply(track_prim)
        rigid_api.CreateRigidBodyEnabledAttr(False)  # Static
        
        # Create cart
        cart_prim = stage.DefinePrim("/World/Cart_0", "Xform")
        cart_geom = UsdGeom.Cube.Define(stage, "/World/Cart_0/Geometry")
        cart_geom.CreateSizeAttr(1.0)
        cart_geom.GetPrim().CreateAttribute("xformOp:scale", Sdf.ValueTypeNames.Float3).Set(Gf.Vec3f(0.4, 0.3, 0.2))
        cart_geom.GetPrim().CreateAttribute("xformOp:translate", Sdf.ValueTypeNames.Double3).Set(Gf.Vec3d(0, 0, 0.15))
        cart_geom.CreateXformOpOrderAttr().Set(["xformOp:translate", "xformOp:scale"])
        
        # Add physics to cart
        UsdPhysics.CollisionAPI.Apply(cart_geom.GetPrim())
        UsdPhysics.RigidBodyAPI.Apply(cart_prim)
        UsdPhysics.MassAPI.Apply(cart_prim).CreateMassAttr(1.0)
        
        # Create pendulum with initial angle offset (15 degrees)
        pendulum_prim = stage.DefinePrim("/World/Pendulum_0", "Xform")
        UsdGeom.Xformable(pendulum_prim).AddTranslateOp().Set(Gf.Vec3d(0, 0, 0.25))
        UsdGeom.Xformable(pendulum_prim).AddRotateYOp().Set(15.0)  # 15 degree rotation around Y axis
        
        pendulum_geom = UsdGeom.Cylinder.Define(stage, "/World/Pendulum_0/Geometry")
        pendulum_geom.CreateRadiusAttr(0.02)
        pendulum_geom.CreateHeightAttr(1.0)
        pendulum_geom.GetPrim().CreateAttribute("xformOp:translate", Sdf.ValueTypeNames.Double3).Set(Gf.Vec3d(0, 0, 0.5))
        pendulum_geom.CreateXformOpOrderAttr().Set(["xformOp:translate"])
        
        # Add physics to pendulum
        UsdPhysics.CollisionAPI.Apply(pendulum_geom.GetPrim())
        UsdPhysics.RigidBodyAPI.Apply(pendulum_prim)
        UsdPhysics.MassAPI.Apply(pendulum_prim).CreateMassAttr(0.1)
        
        # Create prismatic joint for cart
        cart_joint = stage.DefinePrim("/World/CartJoint_0", "PhysicsPrismaticJoint")
        joint_api = UsdPhysics.PrismaticJoint(cart_joint)
        joint_api.CreateBody0Rel().SetTargets(["/World/Track_0"])
        joint_api.CreateBody1Rel().SetTargets(["/World/Cart_0"])
        joint_api.CreateAxisAttr("X")
        joint_api.CreateLowerLimitAttr(-5.0)
        joint_api.CreateUpperLimitAttr(5.0)
        
        # Create revolute joint for pendulum
        pendulum_joint = stage.DefinePrim("/World/PendulumJoint_0", "PhysicsRevoluteJoint")
        joint_api = UsdPhysics.RevoluteJoint(pendulum_joint)
        joint_api.CreateBody0Rel().SetTargets(["/World/Cart_0"])
        joint_api.CreateBody1Rel().SetTargets(["/World/Pendulum_0"])
        joint_api.CreateAxisAttr("Y")

    def get_observations(self) -> dict:
        positions, _ = self._pendulums.get_world_poses()
        velocities = self._pendulums.get_velocities()
        return {"positions": positions, "velocities": velocities}

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
    
    # Save USD file
    stage = omni.usd.get_context().get_stage()
    stage.Export("inverted_pendulum_simple.usd")
    print("Simulation saved to inverted_pendulum_simple.usd")
    
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