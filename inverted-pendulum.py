import warp as wp
import warp.sim
import warp.sim.render
import numpy as np
import math

# Initialize Warp
wp.init()

class CartPoleSimulation:
    def __init__(self, num_envs=1, device="cuda"):
        self.num_envs = num_envs
        self.device = device if wp.is_cuda_available() else "cpu"
        
        # Simulation parameters
        self.dt = 1.0 / 60.0  # 60 FPS
        self.sim_substeps = 10
        self.sim_dt = self.dt / self.sim_substeps
        
        # Physical parameters
        self.cart_mass = 1.0
        self.pole_mass = 0.1
        self.pole_length = 1.0
        self.cart_width = 0.5
        self.cart_height = 0.3
        self.cart_depth = 0.3
        
        # Track parameters
        self.track_length = 10.0
        self.track_width = 0.1
        self.track_height = 0.05
        
        self.builder = None
        self.model = None
        self.states = []
        self.controls = []
        self.renderer = None
        
        self._build_simulation()
    
    def _build_simulation(self):
        """Build the cart pole simulation model"""
        self.builder = wp.sim.ModelBuilder()
        
        # Create environments
        for env_id in range(self.num_envs):
            env_offset = wp.vec3(env_id * 3.0, 0.0, 0.0)  # Spread environments apart
            self._create_cartpole_env(env_id, env_offset)
        
        # Build the model
        self.model = self.builder.finalize(device=self.device)
        self.model.ground = True  # Enable ground collision
        
        # Create initial states for all environments
        for env_id in range(self.num_envs):
            state = self.model.state(requires_grad=True)
            self.states.append(state)
            
            # Set initial random conditions
            self._reset_env_state(state, env_id)
        
        # Initialize controls (joint torques/forces)
        self.controls = [wp.zeros(self.model.joint_count, dtype=wp.float32, device=self.device) 
                        for _ in range(self.num_envs)]
    
    def _create_cartpole_env(self, env_id, offset):
        """Create a single cart pole environment"""
        
        # Create track (static ground)
        track_pos = offset + wp.vec3(0.0, -self.cart_height/2 - self.track_height/2, 0.0)
        
        track_body = self.builder.add_body(
            origin=wp.transform(track_pos, wp.quat_identity()),
            name=f"track_{env_id}"
        )
        self.builder.add_shape_box(
            body=track_body,
            hx=self.track_length/2,
            hy=self.track_height/2,
            hz=self.track_width/2,
            density=1000.0
        )
        
        # Create cart
        cart_pos = offset + wp.vec3(0.0, 0.0, 0.0)
        cart_body = self.builder.add_body(
            origin=wp.transform(cart_pos, wp.quat_identity()),
            name=f"cart_{env_id}"
        )
        self.builder.add_shape_box(
            body=cart_body,
            hx=self.cart_width/2,
            hy=self.cart_height/2,
            hz=self.cart_depth/2,
            density=self.cart_mass / (self.cart_width * self.cart_height * self.cart_depth)
        )
        
        # Create pole
        pole_pos = offset + wp.vec3(0.0, self.pole_length/2, 0.0)
        pole_body = self.builder.add_body(
            origin=wp.transform(pole_pos, wp.quat_identity()),
            name=f"pole_{env_id}"
        )
        self.builder.add_shape_capsule(
            body=pole_body,
            radius=0.02,
            half_height=self.pole_length/2,
            density=self.pole_mass / (math.pi * (0.02**2) * self.pole_length)
        )
        
        # Add revolute joint between cart and pole (hinge at cart center)
        self.builder.add_joint_revolute(
            parent=cart_body,
            child=pole_body,
            parent_xform=wp.transform(wp.vec3(0.0, self.cart_height/2, 0.0), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(0.0, -self.pole_length/2, 0.0), wp.quat_identity()),
            axis=wp.vec3(0.0, 0.0, 1.0)  # Rotation around Z-axis
        )
        
        # Add prismatic joint to constrain cart to horizontal movement
        self.builder.add_joint_prismatic(
            parent=-1,  # World reference
            child=cart_body,
            axis=wp.vec3(1.0, 0.0, 0.0),  # X-axis movement only
            parent_xform=wp.transform(cart_pos, wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
            limit_lower=-2.4,  # Cart position limits
            limit_upper=2.4
        )
        # Store joint index for control
        prismatic_joint_idx = len(self.builder.joint_name) - 1
        
        # Add control for cart force (will be applied to the prismatic joint)
        # Note: Control will be applied via joint forces during simulation
    
    def _reset_env_state(self, state, env_id):
        """Reset environment to initial conditions with some randomization"""
        # Find bodies for this environment
        cart_idx = None
        pole_idx = None
        
        for i, name in enumerate(self.model.body_name):
            if name == f"cart_{env_id}":
                cart_idx = i
            elif name == f"pole_{env_id}":
                pole_idx = i
        
        if cart_idx is not None:
            # Random cart position
            cart_x = np.random.uniform(-0.5, 0.5)
            state.body_q[cart_idx] = wp.vec3(env_id * 3.0 + cart_x, 0.0, 0.0)
            state.body_qd[cart_idx] = wp.vec3(np.random.uniform(-0.1, 0.1), 0.0, 0.0)
        
        if pole_idx is not None:
            # Random pole angle (small perturbation from vertical)
            angle = np.random.uniform(-0.2, 0.2)
            pole_quat = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), angle)
            pole_x = env_id * 3.0 + cart_x
            pole_y = self.pole_length/2 * math.cos(angle)
            state.body_q[pole_idx] = wp.vec3(pole_x, pole_y, 0.0)
            state.body_rot[pole_idx] = pole_quat
            
            # Small initial angular velocity
            state.body_qd[pole_idx] = wp.vec3(0.0, 0.0, 0.0)
            angular_vel = wp.vec3(0.0, 0.0, np.random.uniform(-0.1, 0.1))
            state.body_omega[pole_idx] = angular_vel
    
    def reset(self, env_indices=None):
        """Reset specified environments or all if None"""
        if env_indices is None:
            env_indices = list(range(self.num_envs))
        
        for env_id in env_indices:
            if env_id < len(self.states):
                self._reset_env_state(self.states[env_id], env_id)
    
    def set_forces(self, forces):
        """Set control forces for all environments"""
        if not isinstance(forces, list):
            forces = [forces] * self.num_envs
        
        for env_id in range(min(len(forces), self.num_envs)):
            if env_id < len(self.controls):
                # Find the prismatic joint for this environment (should be the first joint for each env)
                prismatic_joint_idx = env_id * 2  # 2 joints per env (prismatic + revolute)
                if prismatic_joint_idx < len(self.controls[env_id]):
                    self.controls[env_id].numpy()[prismatic_joint_idx] = forces[env_id]
    
    def step(self):
        """Step the simulation forward"""
        for env_id in range(self.num_envs):
            for _ in range(self.sim_substeps):
                wp.sim.collide(self.model, self.states[env_id])
                
                state_prev = self.states[env_id]
                state_next = self.model.state(requires_grad=True)
                
                wp.sim.integrate(
                    self.model, 
                    state_prev, 
                    state_next, 
                    self.sim_dt,
                    joint_forces=self.controls[env_id]  # Apply joint forces
                )
                
                self.states[env_id] = state_next
    
    def get_state(self, env_id=0):
        """Get current state of specified environment"""
        if env_id >= len(self.states):
            return None
        
        state = self.states[env_id]
        
        # Find cart and pole bodies
        cart_idx = pole_idx = None
        for i, name in enumerate(self.model.body_name):
            if name == f"cart_{env_id}":
                cart_idx = i
            elif name == f"pole_{env_id}":
                pole_idx = i
        
        result = {}
        if cart_idx is not None:
            cart_pos = state.body_q[cart_idx].numpy()
            cart_vel = state.body_qd[cart_idx].numpy()
            result.update({
                'cart_pos': cart_pos[0] - env_id * 3.0,  # Remove environment offset
                'cart_vel': cart_vel[0]
            })
        
        if pole_idx is not None:
            pole_rot = state.body_rot[pole_idx].numpy()
            pole_omega = state.body_omega[pole_idx].numpy()
            
            # Convert quaternion to angle (assuming rotation around Z-axis)
            # For small angles, we can approximate
            angle = 2.0 * math.atan2(pole_rot[2], pole_rot[3])  # Extract Z rotation from quaternion
            
            result.update({
                'pole_angle': angle,
                'pole_angular_vel': pole_omega[2]
            })
        
        return result
    
    def is_done(self, env_id=0):
        """Check if episode should terminate"""
        state = self.get_state(env_id)
        if state is None:
            return True
        
        # Termination conditions
        cart_limit = 2.4
        angle_limit = 12 * math.pi / 180  # 12 degrees in radians
        
        cart_pos = state.get('cart_pos', 0)
        pole_angle = state.get('pole_angle', 0)
        
        return (abs(cart_pos) > cart_limit or abs(pole_angle) > angle_limit)
    
    def create_renderer(self, camera_pos=None, camera_target=None):
        """Create a renderer for visualization"""
        if camera_pos is None:
            camera_pos = (5.0, 5.0, 5.0)
        if camera_target is None:
            camera_target = (0.0, 0.0, 0.0)
            
        self.renderer = wp.sim.render.SimRenderer(
            self.model, 
            stage_path="cartpole.usd",
            scaling=1.0
        )
        
        return self.renderer
    
    def render(self, env_id=0):
        """Render the specified environment"""
        if self.renderer is None:
            self.create_renderer()
        
        if env_id < len(self.states):
            self.renderer.begin_frame(time=0.0)
            self.renderer.render(self.states[env_id])
            self.renderer.end_frame()

# Example usage with simple control loop
def main():
    print("Cart Pole Simulation with Warp.sim")
    print("==================================")
    
    # Create simulation
    sim = CartPoleSimulation(num_envs=2)
    
    # Create renderer (optional)
    try:
        renderer = sim.create_renderer()
        has_renderer = True
        print("Renderer created successfully")
    except Exception as e:
        print(f"Could not create renderer: {e}")
        has_renderer = False
    
    # Simple PD controller for demonstration
    def pd_controller(state, target_angle=0.0, kp=20.0, kd=5.0):
        """Simple PD controller to balance the pole"""
        angle_error = target_angle - state['pole_angle']
        angular_vel_error = -state['pole_angular_vel']
        
        force = kp * angle_error + kd * angular_vel_error
        return np.clip(force, -10.0, 10.0)  # Limit force
    
    # Run simulation
    max_steps = 1000
    
    for step in range(max_steps):
        # Get current states
        states = [sim.get_state(i) for i in range(sim.num_envs)]
        
        # Compute control forces
        forces = []
        for env_id in range(sim.num_envs):
            state = states[env_id]
            if state:
                if env_id == 0:
                    # First environment: PD controller
                    force = pd_controller(state)
                else:
                    # Second environment: oscillating force
                    force = 5.0 * math.sin(step * 0.1)
                forces.append(force)
            else:
                forces.append(0.0)
        
        # Apply forces and step simulation
        sim.set_forces(forces)
        sim.step()
        
        # Print status every 60 steps
        if step % 60 == 0:
            print(f"\nStep {step}:")
            for env_id in range(sim.num_envs):
                state = sim.get_state(env_id)
                done = sim.is_done(env_id)
                if state:
                    print(f"  Env {env_id}: Cart={state['cart_pos']:.2f}, "
                          f"Angle={state['pole_angle']:.2f}, Done={done}")
                
                # Reset if done
                if done:
                    print(f"  Resetting environment {env_id}")
                    sim.reset([env_id])
        
        # Render first environment
        if has_renderer and step % 10 == 0:
            sim.render(0)

if __name__ == "__main__":
    main()