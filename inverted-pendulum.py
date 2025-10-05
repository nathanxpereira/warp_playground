import numpy as np
import math
import newton
import warp as wp
# Initialize Warp
wp.init()

class CartPoleSimulation:
    def __init__(self, num_envs=1, device="cuda"):
        self.num_envs = num_envs
        self.device = device if wp.is_cuda_available() else "cpu"
        
        # Environment parameters
        self.env_spacing = 3.0  # Space between environments

        # Simulation parameters
        self.dt = 1.0 / 60.0  # 60 FPS
        self.sim_substeps = 1
        self.sim_dt = self.dt / self.sim_substeps
        
        # Physical parameters
        self.cart_mass = 1.0
        self.pole_mass = 0.1
        self.pole_length = 1.0
        self.pole_radius = 0.02  # Radius of the pole
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
        self.builder = newton.ModelBuilder()
        
        # Create environments
        for env_id in range(self.num_envs):
            env_offset = wp.vec3(env_id * self.env_spacing, 0.0, 0.0)  # Spread environments apart
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
        self.controls = [wp.zeros(int(self.model.joint_count/self.num_envs), dtype=wp.float32, device=self.device) 
                        for _ in range(self.num_envs)]
        
        self.integrator = newton.SemiImplicitIntegrator()

    
    def _create_cartpole_env(self, env_id, offset):
        """Create a single cart pole environment"""
        
        # Create track (static ground)
        track_pos = offset + wp.vec3(0.0, -self.cart_height/2 - self.track_height/2, 0.0)
        
        # Create a static body for the track
        track_body = self.builder.add_body(
            origin=wp.transform(track_pos, wp.quat_identity()),
            name=f"track_{env_id}"
        )
        # input half extents for a box shape
        # hx, hy, hz are half extents along x, y, z axes respectively
        # density is mass per unit volume
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
        # pole is positioned at the top of the cart
        # so we need to adjust the position based on cart height and pole length
        pole_pos = offset + wp.vec3(0.0, self.cart_height/2+self.pole_length/2, 0.0)
        pole_body = self.builder.add_body(
            origin=wp.transform(pole_pos, wp.quat_identity()),
            name=f"pole_{env_id}"
        )
        self.builder.add_shape_capsule(
            body=pole_body,
            radius=0.02,
            half_height=self.pole_length/2,
            density=self.pole_mass / (math.pi * (self.pole_radius**2) * self.pole_length)
        )
        
        # the position of the joint on the parent is relative to the parent's origin
        # and the position of the joint on the child is relative to the child's origin
        # Here we create a revolute joint between the cart and the pole
        # The joint allows the pole to rotate around the cart's center
        # around the Z-axis (vertical axis)
        self.builder.add_joint_revolute(
            parent=cart_body,
            child=pole_body,
            parent_xform=wp.transform(wp.vec3(0.0, self.cart_height/2, 0.0), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(0.0, -self.pole_length/2, 0.0), wp.quat_identity()),
            axis=wp.vec3(0.0, 0.0, 1.0),  # Rotation around Z-axis
            name='revolute_joint_' + str(env_id),
        )
        
        # Add prismatic joint to constrain cart to horizontal movement
        self.builder.add_joint_prismatic(
            parent=-1,  # World reference
            child=cart_body,
            axis=wp.vec3(1.0, 0.0, 0.0),  # X-axis movement only
            parent_xform=wp.transform(cart_pos, wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
            limit_lower=-2.4,  # Cart position limits
            limit_upper=2.4,
            name='prismatic_joint_' + str(env_id)
        )
        # Store joint index for control
        self.prismatic_joint_idx = len(self.builder.joint_name) - 1
        
        # Add control for cart force (will be applied to the prismatic joint)
        # Note: Control will be applied via joint forces during simulation
    
    def _reset_env_state(self, state: newton.State, env_id):
        """Reset environment to initial conditions with some randomization"""
        # Find bodies for this environment
        cart_idx = None
        pole_idx = None
        
        for i, name in enumerate(self.model.body_name):
            if name == f"cart_{env_id}":
                cart_idx = i
            elif name == f"pole_{env_id}":
                pole_idx = i
        
        # What is state.body_q, state.body_qd, state.body_rot, state.body_omega?
        # state.body_q: positions of bodies (x,y,z) + quaternion coords (x,y,z,w) ==> (num_bodies,7)
        # state.body_qd: spatial and angular velocities of bodies (num_bodies,6)

        body_q = state.body_q.numpy()
        body_qd = state.body_qd.numpy()

        if cart_idx is not None:
            # Random cart position
            cart_x = np.random.uniform(-0.5, 0.5)
            cart_pos = wp.vec3(env_id * self.env_spacing + cart_x, 0.0, 0.0)
            cart_vel = np.array([np.random.uniform(-0.1, 0.1), 0.0, 0.0, 0.0, 0.0, 0.0])  # No initial angular velocity
            body_q[cart_idx,:]  = np.array(wp.transform(cart_pos, wp.quat_identity()))
            body_qd[cart_idx,:] = cart_vel
            
        
        if pole_idx is not None:
            # Random pole angle (small perturbation from vertical)
            angle = np.random.uniform(-0.2, 0.2)
            pole_quat = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), angle)
            pole_x = env_id * self.env_spacing + cart_x + self.pole_length/2 * math.sin(angle)
            pole_y = self.pole_length/2 * math.cos(angle)
            pole_pos = wp.vec3(pole_x, pole_y, 0.0)
            body_q[pole_idx] = np.array(wp.transform(pole_pos, pole_quat))
            
            # Small initial angular velocity
            body_qd[pole_idx] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, np.random.uniform(-0.1, 0.1)])  # Pole angular velocity around Z-axis
        
        # Update the state with the new positions and velocities
        state.body_q = wp.array(body_q, dtype=wp.transform, device=self.device)
        state.body_qd = wp.array(body_qd, dtype=wp.spatial_vectorf, device=self.device)
    
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
        
        controls = [control.numpy() for control in self.controls]
        for env_id in range(min(len(forces), self.num_envs)):
            if env_id < len(controls):
                # Find the prismatic joint for this environment (should be the first joint for each env)
                prismatic_joint_idx = np.where(np.array(self.model.joint_name) == f"prismatic_joint_{env_id}")[0]
                if prismatic_joint_idx < len(controls[env_id]):
                    controls[env_id][prismatic_joint_idx] = forces[env_id]
        
        self.controls = [wp.array(control, dtype=wp.float32, device=self.device) for control in controls]
    
    def step(self):
        """Step the simulation forward"""
        for env_id in range(self.num_envs):
            for _ in range(self.sim_substeps):
                newton.collide(self.model, self.states[env_id])
                
                self.state_prev = self.states[env_id]
                self.state_next = self.model.state(requires_grad=True)
                
                # figure out why this no worky
                self.integrator.simulate(
                    model=self.model, 
                    state_in=self.state_prev, 
                    state_out=self.state_next, 
                    dt=self.sim_dt,
                    control=self.model.control(self.controls[env_id])  # Apply joint forces
                )
                
                self.states[env_id] = self.state_next
    
    def get_state(self, env_id=0):
        """Get current state of specified environment"""
        if env_id >= len(self.states):
            return None
        
        state:newton.State = self.states[env_id]
        
        # Find cart and pole bodies
        cart_idx = pole_idx = None
        for i, name in enumerate(self.model.body_name):
            if name == f"cart_{env_id}":
                cart_idx = i
            elif name == f"pole_{env_id}":
                pole_idx = i
        
        result = {}
        if cart_idx is not None:
            cart_pos = state.body_q.numpy()[cart_idx]
            cart_vel = state.body_qd.numpy()[cart_idx]
            result.update({
                'cart_pos': cart_pos[0] - env_id * self.env_spacing,  # Remove environment offset
                'cart_vel': cart_vel[0]
            })
        
        if pole_idx is not None:
            pole_rot = state.body_q.numpy()[pole_idx,-4:]
            pole_omega = state.body_qd.numpy()[pole_idx,-3:]
            
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
            
        self.renderer = newton.render.SimRenderer(
            model=self.model, 
            path="cartpole.usd",
            up_axis='Y',
            fps=1/self.dt,
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
    print("Cart Pole Simulation with newton")
    print("==================================")
    
    # Create simulation
    sim = CartPoleSimulation(num_envs=2)
    
    # Create renderer (optional)
    renderer = sim.create_renderer()
    has_renderer = True
    print("Renderer created successfully")
    # print(f"Could not create renderer: {e}")
    # has_renderer = False
    
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
        
        # Compute control forces for each environment
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
        if step % 1 == 0:
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
        if has_renderer and step % 1 == 0:
            sim.render(0)

if __name__ == "__main__":
    main()