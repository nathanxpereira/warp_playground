from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import numpy as np
import torch
from isaacsim.core.api import World
from isaacsim.core.api.tasks import BaseTask
from isaacsim.core.prims import RigidPrim
import omni.usd
from pxr import Gf, UsdPhysics, UsdGeom, Sdf
from controller import *

class CartPole(BaseTask):
    def __init__(self, name, num_envs, offset=None) -> None:
        BaseTask.__init__(self, name=name, offset=offset)
        self._num_envs = num_envs

    def set_up_scene(self, scene) -> None:
        super().set_up_scene(scene)
        scene.add_default_ground_plane()
        self._create_cartpole()
        
        self._carts = RigidPrim("/World/Cart_0", "cart")
        self._poles = RigidPrim("/World/Pole_0", "pole")
        scene.add(self._carts)
        scene.add(self._poles)

    def _create_cartpole(self):
        stage = omni.usd.get_context().get_stage()
        
        # Cart
        cart_prim = stage.DefinePrim("/World/Cart_0", "Xform")
        cart_geom = UsdGeom.Cube.Define(stage, "/World/Cart_0/Geom")
        cart_geom.CreateSizeAttr(1.0)
        cart_geom.GetPrim().CreateAttribute("xformOp:scale", Sdf.ValueTypeNames.Float3).Set(Gf.Vec3f(0.4, 0.3, 0.2))
        cart_geom.GetPrim().CreateAttribute("xformOp:translate", Sdf.ValueTypeNames.Double3).Set(Gf.Vec3d(0, 0, 0.1))
        cart_geom.CreateXformOpOrderAttr().Set(["xformOp:translate", "xformOp:scale"])
        UsdPhysics.CollisionAPI.Apply(cart_geom.GetPrim())
        UsdPhysics.RigidBodyAPI.Apply(cart_prim)
        UsdPhysics.MassAPI.Apply(cart_prim).CreateMassAttr(1.0)
        
        # Pole with 15 degree initial angle
        pole_prim = stage.DefinePrim("/World/Pole_0", "Xform")
        UsdGeom.Xformable(pole_prim).AddTranslateOp().Set(Gf.Vec3d(0, 0, 0.2))
        UsdGeom.Xformable(pole_prim).AddRotateYOp().Set(15.0)
        
        pole_geom = UsdGeom.Cylinder.Define(stage, "/World/Pole_0/Geom")
        pole_geom.CreateRadiusAttr(0.02)
        pole_geom.CreateHeightAttr(1.0)
        pole_geom.GetPrim().CreateAttribute("xformOp:translate", Sdf.ValueTypeNames.Double3).Set(Gf.Vec3d(0, 0, 0.5))
        pole_geom.CreateXformOpOrderAttr().Set(["xformOp:translate"])
        UsdPhysics.CollisionAPI.Apply(pole_geom.GetPrim())
        UsdPhysics.RigidBodyAPI.Apply(pole_prim)
        UsdPhysics.MassAPI.Apply(pole_prim).CreateMassAttr(0.1)
        
        # Prismatic joint - cart constrained to X-axis
        cart_joint = stage.DefinePrim("/World/CartJoint", "PhysicsPrismaticJoint")
        joint = UsdPhysics.PrismaticJoint(cart_joint)
        joint.CreateBody1Rel().SetTargets(["/World/Cart_0"])
        joint.CreateAxisAttr("X")
        joint.CreateLowerLimitAttr(-5.0)
        joint.CreateUpperLimitAttr(5.0)
        
        # Revolute joint - pole pivots on cart
        pole_joint = stage.DefinePrim("/World/PoleJoint", "PhysicsRevoluteJoint")
        joint = UsdPhysics.RevoluteJoint(pole_joint)
        joint.CreateBody0Rel().SetTargets(["/World/Cart_0"])
        joint.CreateBody1Rel().SetTargets(["/World/Pole_0"])
        joint.CreateLocalPos0Attr(Gf.Vec3f(0, 0, 0.2))  # Top of cart
        joint.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))    # Bottom of pole (at pole origin)
        joint.CreateAxisAttr("Y")

    def get_observations(self) -> dict:
        cart_pos, _ = self._carts.get_world_poses()
        pole_pos, pole_rot = self._poles.get_world_poses()
        cart_vel = self._carts.get_velocities()
        pole_vel = self._poles.get_velocities()
        return {
            "cart_position": cart_pos,
            "pole_position": pole_pos,
            "pole_rotation": pole_rot,
            "cart_velocity": cart_vel,
            "pole_velocity": pole_vel
        }
    
    def apply_control(self, force):
        """Apply force to cart in X direction"""
        forces = torch.zeros((self._num_envs, 3), device=self._device)
        forces[:, 0] = force  # X-axis force
        self._carts.apply_forces(forces)

    def calculate_metrics(self) -> None:
        return torch.zeros(self._num_envs, device=self._device)

    def is_done(self) -> None:
        return torch.zeros(self._num_envs, device=self._device)



def main():
    my_world = World(stage_units_in_meters=1.0, physics_prim_path="/physicsScene", backend="torch", device="cuda:0")
    my_task = CartPole(name="cartpole", num_envs=1)
    my_world.add_task(my_task)
    my_world.reset()
    
    stage = omni.usd.get_context().get_stage()
    stage.Export("cartpole.usd")
    print("Simulation saved to cartpole.usd")
    
    controller = LQRController()  # or PIDController()
    
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
                force = controller.compute_control(observations, my_world.device)
                my_task.apply_control(force)
                
            my_world.step(render=True)
    finally:
        my_world.stop()
        simulation_app.close()

if __name__ == "__main__":
    main()
