import numpy as np
import plotly.express as px
import polars as pl

# import physics config to generate A and B for example
from Cartpole.config import CartPolePhysicsConfig
from Cartpole.controllers import LQRController

if __name__=="__main__":
    Q = np.diag([0.1, 0.0001, 0.1, 1.0])  # State cost weights [cart_pos, cart_vel, pole_angle, pole_vel]
    R = np.array([[0.1]])  # Control cost weight
    x0 = np.array([[0,0,np.deg2rad(1.0),0]]).T
    cfg = CartPolePhysicsConfig(cart_mass=1.0, pole_mass=1.0, pole_length=1.0, pole_diam=0.6)
    A, B = cfg.compute_system_matrices()
    c = LQRController(A,B,Q,R)
    
    t,x,u = c.simulate_closed_loop(x0,duration=10)

    print(c.is_stable())
    df = pl.DataFrame({'t': t, 'pos': x[0,:],'angle': x[2,:], 'ang_vel': x[3,:]})

    # 2. Create an interactive scatter plot
    fig = px.line(
        df, 
        x="t", 
        y="pos", 
        title="Interactive Iris Dataset Scatter Plot"
    )

    # 3. Render and display the plot in your browser or notebook
    fig.show()
    
    

