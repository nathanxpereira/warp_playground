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

        self.cart_length = 0.4
        self.cart_width = 0.3
        self.cart_height = 0.2
        self.cart_mass = 1.0

        self.pole_radius = 0.02
        self.pole_height = 1.0
        self.pole_mass = 0.1
        self.start_angle = 5.0

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
        cart_geom.GetPrim().CreateAttribute("xformOp:scale", Sdf.ValueTypeNames.Float3).Set(Gf.Vec3f(self.cart_length, self.cart_width, self.cart_height))
        cart_geom.GetPrim().CreateAttribute("xformOp:translate", Sdf.ValueTypeNames.Double3).Set(Gf.Vec3d(0, 0, self.cart_height/2))
        cart_geom.CreateXformOpOrderAttr().Set(["xformOp:translate", "xformOp:scale"])
        UsdPhysics.CollisionAPI.Apply(cart_geom.GetPrim())
        UsdPhysics.RigidBodyAPI.Apply(cart_prim)
        UsdPhysics.MassAPI.Apply(cart_prim).CreateMassAttr(self.cart_mass)
        
        # Pole with initial angle
        pole_prim = stage.DefinePrim("/World/Pole_0", "Xform")
        UsdGeom.Xformable(pole_prim).AddTranslateOp().Set(Gf.Vec3d(0, 0, self.cart_height))
        UsdGeom.Xformable(pole_prim).AddRotateYOp().Set(self.start_angle)
        
        pole_geom = UsdGeom.Cylinder.Define(stage, "/World/Pole_0/Geom")
        pole_geom.CreateRadiusAttr(self.pole_radius)
        pole_geom.CreateHeightAttr(self.pole_height)
        pole_geom.GetPrim().CreateAttribute("xformOp:translate", Sdf.ValueTypeNames.Double3).Set(Gf.Vec3d(0, 0, self.pole_height/2))
        pole_geom.CreateXformOpOrderAttr().Set(["xformOp:translate"])
        UsdPhysics.CollisionAPI.Apply(pole_geom.GetPrim())
        UsdPhysics.RigidBodyAPI.Apply(pole_prim)
        UsdPhysics.MassAPI.Apply(pole_prim).CreateMassAttr(self.pole_mass)
        
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
        joint.CreateLocalPos0Attr(Gf.Vec3f(0, 0, self.cart_height))  # Top of cart
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
    
    def reset_state(self):
        angle_rad = np.radians(self.start_angle)
        pole_quat = torch.tensor([[np.cos(angle_rad/2), 0, np.sin(angle_rad/2), 0]], dtype=torch.float32, device=self._device)
        self._carts.set_world_poses(torch.tensor([[0, 0, self.cart_height/2]], dtype=torch.float32, device=self._device))
        self._poles.set_world_poses(torch.tensor([[0, 0, self.cart_height]], dtype=torch.float32, device=self._device), pole_quat)
        self._carts.set_velocities(torch.zeros((1, 6), dtype=torch.float32, device=self._device))
        self._poles.set_velocities(torch.zeros((1, 6), dtype=torch.float32, device=self._device))



def main():
    my_world = World(stage_units_in_meters=1.0, physics_prim_path="/physicsScene", backend="torch", device="cuda:0")
    my_task = CartPole(name="cartpole", num_envs=1)
    my_world.add_task(my_task)
    my_world.reset()
    my_world.pause()
    
    stage = omni.usd.get_context().get_stage()
    stage.Export("cartpole/cartpole.usd")
    print("Simulation saved to cartpole.usd")
    
    controller = LQRController(m_cart=my_task.cart_mass, m_pole=my_task.pole_mass, l=my_task.pole_height)
    
    reset_needed = True
    try:
        while simulation_app.is_running():
            if my_world.is_stopped() and not reset_needed:
                reset_needed = True
            if my_world.is_playing():
                if reset_needed:
                    my_world.reset(soft=True)
                    my_task.reset_state()
                    for _ in range(3):
                        my_world.step(render=False)
                    controller.reset()
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
