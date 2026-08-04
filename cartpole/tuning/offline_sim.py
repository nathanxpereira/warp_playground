import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import solve_continuous_are

# import physics config to generate A and B for example
from Cartpole.config import CartPolePhysicsConfig

def compute_gain(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray) -> np.ndarray:
    # what is P again? 
    P = solve_continuous_are(A, B, Q, R)
    K = np.linalg.solve(R, B.T @ P) 
    return K

def simulate_closed_loop(
    A: np.ndarray,
    B: np.ndarray,
    K: np.ndarray,
    x0: np.ndarray,
    dt: float = 1 / 240,
    duration: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Integrate dx/dt = (A - B@K) x from x0 and recover u(t) = -K @ x(t).
    Returns:
        t: (N,) time points
        x: (N,4) state trajectory
        u: (N,) control force trajectory
    """
    lqr_sys = lambda t,x: (A-B@K)@x
    sol = solve_ivp(lqr_sys, t_span=[0,duration], y0=x0.ravel(), t_eval=np.arange(0,duration,dt))
    t = sol.t
    x = sol.y
    u = -K@x
    return t,x,u

def is_stable(A: np.ndarray, B: np.ndarray, K: np.ndarray) -> bool:
    eigs = np.linalg.eig(A-B@K)
    print(eigs.eigenvalues)
    return np.all(eigs.eigenvalues<0)

if __name__=="__main__":
    Q = np.diag([1.0, 1.0, 10.0, 10.0])  # State cost weights [cart_pos, cart_vel, pole_angle, pole_vel]
    R = np.array([[0.1]])  # Control cost weight
    x0 = np.array([[0,0,np.deg2rad(2.0),0]]).T
    cfg = CartPolePhysicsConfig(cart_mass=1.0, pole_mass=1.0, pole_length=1.0)
    A, B = cfg.compute_system_matrices()
    K = compute_gain(A, B, Q, R)
    
    t,x,u = simulate_closed_loop(A, B, K, x0)
    print(is_stable(A, B, K))

