import warp as wp
import numpy as np
import math
import time

# Initialize Warp
wp.init()

@wp.kernel
def simulate_cartpole(
    cart_pos: wp.array(dtype=float),
    cart_vel: wp.array(dtype=float),
    pole_angle: wp.array(dtype=float),
    pole_vel: wp.array(dtype=float),
    force: wp.array(dtype=float),
    dt: float,
    gravity: float,
    mass_cart: float,
    mass_pole: float,
    length: float,
    pole_mass_length: float
):
    tid = wp.tid()
    
    # Current state
    x = cart_pos[tid]
    x_dot = cart_vel[tid]
    theta = pole_angle[tid]
    theta_dot = pole_vel[tid]
    f = force[tid]
    
    # Physics calculations
    cos_theta = wp.cos(theta)
    sin_theta = wp.sin(theta)
    
    # Temporary variables for dynamics
    temp = (f + pole_mass_length * theta_dot * theta_dot * sin_theta) / (mass_cart + mass_pole)
    theta_acc_num = gravity * sin_theta - cos_theta * temp
    theta_acc_den = length * (4.0/3.0 - mass_pole * cos_theta * cos_theta / (mass_cart + mass_pole))
    theta_acc = theta_acc_num / theta_acc_den
    
    x_acc = temp - pole_mass_length * theta_acc * cos_theta / (mass_cart + mass_pole)
    
    # Update velocities and positions using Euler integration
    x_dot = x_dot + dt * x_acc
    x = x + dt * x_dot
    theta_dot = theta_dot + dt * theta_acc
    theta = theta + dt * theta_dot
    
    # Update arrays
    cart_pos[tid] = x
    cart_vel[tid] = x_dot
    pole_angle[tid] = theta
    pole_vel[tid] = theta_dot

@wp.kernel
def update_vertices(
    cart_pos: wp.array(dtype=float),
    pole_angle: wp.array(dtype=float),
    vertices: wp.array(dtype=wp.vec3),
    pole_length: float,
    cart_width: float,
    cart_height: float
):
    tid = wp.tid()
    
    x = cart_pos[tid]
    theta = pole_angle[tid]
    
    # Cart vertices (rectangular prism)
    hw = cart_width * 0.5
    hh = cart_height * 0.5
    
    # Cart vertices
    vertices[tid * 14 + 0] = wp.vec3(x - hw, -hh, -hw)  # Bottom face
    vertices[tid * 14 + 1] = wp.vec3(x + hw, -hh, -hw)
    vertices[tid * 14 + 2] = wp.vec3(x + hw, -hh, hw)
    vertices[tid * 14 + 3] = wp.vec3(x - hw, -hh, hw)
    vertices[tid * 14 + 4] = wp.vec3(x - hw, hh, -hw)   # Top face
    vertices[tid * 14 + 5] = wp.vec3(x + hw, hh, -hw)
    vertices[tid * 14 + 6] = wp.vec3(x + hw, hh, hw)
    vertices[tid * 14 + 7] = wp.vec3(x - hw, hh, hw)
    
    # Pole vertices (line from cart center to pole tip)
    pole_tip_x = x + pole_length * wp.sin(theta)
    pole_tip_y = pole_length * wp.cos(theta)
    
    vertices[tid * 14 + 8] = wp.vec3(x, 0.0, 0.0)  # Pole base (cart center)
    vertices[tid * 14 + 9] = wp.vec3(pole_tip_x, pole_tip_y, 0.0)  # Pole tip
    
    # Track vertices (ground line)
    track_length = 10.0
    vertices[tid * 14 + 10] = wp.vec3(-track_length, -cart_height, 0.0)
    vertices[tid * 14 + 11] = wp.vec3(track_length, -cart_height, 0.0)
    
    # Pole bob (small sphere at tip represented as point)
    vertices[tid * 14 + 12] = wp.vec3(pole_tip_x, pole_tip_y, 0.0)
    vertices[tid * 14 + 13] = wp.vec3(pole_tip_x, pole_tip_y, 0.0)

class CartPoleRenderer:
    def __init__(self, num_envs=1):
        self.num_envs = num_envs
        self.device = "cuda" if wp.is_cuda_available() else "cpu"
        
        # Physical parameters
        self.gravity = 9.8
        self.mass_cart = 1.0
        self.mass_pole = 0.1
        self.length = 0.5  # Half pole length
        self.pole_mass_length = self.mass_pole * self.length
        self.dt = 0.02
        
        # Rendering parameters
        self.cart_width = 0.5
        self.cart_height = 0.3
        
        # Initialize state arrays
        self.cart_pos = wp.zeros(num_envs, dtype=float, device=self.device)
        self.cart_vel = wp.zeros(num_envs, dtype=float, device=self.device)
        self.pole_angle = wp.zeros(num_envs, dtype=float, device=self.device)
        self.pole_vel = wp.zeros(num_envs, dtype=float, device=self.device)
        self.force = wp.zeros(num_envs, dtype=float, device=self.device)
        
        # Vertex buffer (14 vertices per environment: 8 cart + 2 pole + 2 track + 2 pole bob)
        self.vertices = wp.zeros(num_envs * 14, dtype=wp.vec3, device=self.device)
        
        # Initialize random starting conditions
        self.reset()
    
    def reset(self, env_indices=None):
        """Reset environments to random initial states"""
        if env_indices is None:
            env_indices = list(range(self.num_envs))
        
        for i in env_indices:
            # Random initial conditions
            self.cart_pos.numpy()[i] = np.random.uniform(-0.5, 0.5)
            self.cart_vel.numpy()[i] = np.random.uniform(-0.5, 0.5)
            self.pole_angle.numpy()[i] = np.random.uniform(-0.2, 0.2)
            self.pole_vel.numpy()[i] = np.random.uniform(-0.5, 0.5)
    
    def set_force(self, forces):
        """Set forces for all environments"""
        if isinstance(forces, (int, float)):
            forces = [forces] * self.num_envs
        
        for i in range(self.num_envs):
            self.force.numpy()[i] = forces[i]
    
    def step(self):
        """Step the simulation forward"""
        wp.launch(
            kernel=simulate_cartpole,
            dim=self.num_envs,
            inputs=[
                self.cart_pos,
                self.cart_vel,
                self.pole_angle,
                self.pole_vel,
                self.force,
                self.dt,
                self.gravity,
                self.mass_cart,
                self.mass_pole,
                self.length,
                self.pole_mass_length
            ],
            device=self.device
        )
        
        # Update vertex positions for rendering
        wp.launch(
            kernel=update_vertices,
            dim=self.num_envs,
            inputs=[
                self.cart_pos,
                self.pole_angle,
                self.vertices,
                self.length * 2.0,  # Full pole length
                self.cart_width,
                self.cart_height
            ],
            device=self.device
        )
    
    def get_state(self, env_id=0):
        """Get current state of a specific environment"""
        return {
            'cart_pos': self.cart_pos.numpy()[env_id],
            'cart_vel': self.cart_vel.numpy()[env_id],
            'pole_angle': self.pole_angle.numpy()[env_id],
            'pole_vel': self.pole_vel.numpy()[env_id]
        }
    
    def get_vertices(self):
        """Get current vertex positions for rendering"""
        return self.vertices.numpy()
    
    def is_done(self, env_id=0):
        """Check if episode is done (pole fallen or cart out of bounds)"""
        state = self.get_state(env_id)
        pole_angle_threshold = 12 * 2 * math.pi / 360  # 12 degrees
        cart_pos_threshold = 2.4
        
        return (abs(state['cart_pos']) > cart_pos_threshold or 
                abs(state['pole_angle']) > pole_angle_threshold)

# Simple ASCII renderer for terminal output
class ASCIIRenderer:
    def __init__(self, width=80, height=20):
        self.width = width
        self.height = height
    
    def render(self, renderer, env_id=0):
        state = renderer.get_state(env_id)
        
        # Create display buffer
        display = [[' ' for _ in range(self.width)] for _ in range(self.height)]
        
        # Draw track
        track_y = self.height - 3
        for x in range(self.width):
            display[track_y][x] = '-'
        
        # Cart position mapping
        cart_x = int((state['cart_pos'] + 3) * self.width / 6)  # Map [-3,3] to [0,width]
        cart_x = max(2, min(self.width-3, cart_x))
        
        # Draw cart
        if 0 <= cart_x < self.width:
            display[track_y-1][cart_x-1] = '['
            display[track_y-1][cart_x] = '='
            display[track_y-1][cart_x+1] = ']'
        
        # Draw pole
        pole_length = 8  # Display length
        pole_tip_x = cart_x + int(pole_length * math.sin(state['pole_angle']))
        pole_tip_y = track_y - 1 - int(pole_length * math.cos(state['pole_angle']))
        
        # Draw pole line
        steps = max(abs(pole_tip_x - cart_x), abs(pole_tip_y - (track_y-1)))
        if steps > 0:
            for i in range(steps + 1):
                t = i / steps if steps > 0 else 0
                px = int(cart_x + t * (pole_tip_x - cart_x))
                py = int((track_y-1) + t * (pole_tip_y - (track_y-1)))
                
                if 0 <= px < self.width and 0 <= py < self.height:
                    display[py][px] = '|' if abs(pole_tip_x - cart_x) < abs(pole_tip_y - (track_y-1)) else '-'
        
        # Draw pole tip
        if 0 <= pole_tip_x < self.width and 0 <= pole_tip_y < self.height:
            display[pole_tip_y][pole_tip_x] = 'o'
        
        # Print display
        print('\n' * 2)
        for row in display:
            print(''.join(row))
        
        # Print state info
        print(f"\nCart Pos: {state['cart_pos']:.2f}, Cart Vel: {state['cart_vel']:.2f}")
        print(f"Pole Angle: {state['pole_angle']:.2f} rad, Pole Vel: {state['pole_vel']:.2f}")
        print(f"Done: {renderer.is_done(env_id)}")

# Example usage and demo
def main():
    print("Cart Pole Simulation with NVIDIA Warp")
    print("=====================================")
    
    # Create renderer
    num_envs = 4
    cart_pole = CartPoleRenderer(num_envs)
    ascii_renderer = ASCIIRenderer()
    
    # Run simulation
    max_steps = 1000
    forces = [10.0, -10.0, 0.0, 5.0]  # Different forces for each environment
    
    for step in range(max_steps):
        # Apply different control forces
        cart_pole.set_force(forces)
        
        # Step simulation
        cart_pole.step()
        
        # Render first environment every 10 steps
        if step % 10 == 0:
            print(f"\n=== Step {step} ===")
            ascii_renderer.render(cart_pole, env_id=0)
            
            # Check if any environment is done
            done_envs = [i for i in range(num_envs) if cart_pole.is_done(i)]
            if done_envs:
                print(f"Environments {done_envs} finished!")
                cart_pole.reset(done_envs)
        
        time.sleep(0.05)  # Small delay for visualization

if __name__ == "__main__":
    main()