# CartPole System Mathematical Derivation

## System Description

The cartpole system consists of:
- **Cart**: Mass $m_c$ in kg, constrained to move horizontally along the x-axis, with height $h$. The XY plane intersects the cart midway ($\frac{h}{2}$)
- **Pole**: Mass $m_p$ in kg, length $l$ in m, connected to cart via revolute joint
- **Control Input**: Horizontal force $F$ applied along the x axis to the cart

## Coordinate System

- $x$ - Cart position (horizontal along x-axis)
- $\theta$ - Pole angle from vertical (positive clockwise)
- $\dot{x}$ - Cart velocity
- $\dot{\theta}$ - Pole angular velocity

## Lagrangian Mechanics Approach

### Step 1: Define Position Vectors

**Cart** \
Position: 
$$(x, 0)$$
Velocity:
$$(\dot{x},0)$$
Pole center of mass position:
$$
\begin{align*}
x_p &= x + \frac{l}{2}\sin\theta \\
y_p &= \frac{l}{2}\cos\theta+\frac{h}{2}
\end{align*}
$$
Pole velocity components:
$$
\begin{align*}
\dot{x}_p &= \dot{x} + \frac{l}{2}\cos\theta \cdot \dot{\theta} \\
\dot{y}_p &= -\frac{l}{2}\sin\theta \cdot \dot{\theta}
\end{align*}
$$

### Step 2: Solve for Kinetic and Potential Energy

Note for a rotating pole using König's theorem, the moment of inertia about the center of mass is:  $I_{cm} = \frac{1}{12}m_p l^2$ (uniform rod about center) \

Cart kinetic energy:
$$T_c = \frac{1}{2}m_c\dot{x}^2$$

Pole kinetic energy (translational + rotational):
$$
\begin{align*}
T_p &= \frac{1}{2}m_p(\dot{x}_p^2 + \dot{y}_p^2) + \frac{1}{2}I_{cm}\dot{\theta}^2 \\
&= \frac{1}{2}m_p\left[(\dot{x} + \frac{l}{2}\cos\theta \cdot \dot{\theta})^2 + (-\frac{l}{2}\sin\theta \cdot \dot{\theta})^2\right] + \frac{1}{24}m_p l^2\dot{\theta}^2 \\
&= \frac{1}{2}m_p\dot{x}^2 + \frac{1}{2}m_p l\dot{x}\dot{\theta}\cos\theta + \frac{1}{6}m_p l^2\dot{\theta}^2
\end{align*}
$$

Total kinetic energy:
$$T = T_c + T_p = \frac{1}{2}(m_c + m_p)\dot{x}^2 + \frac{1}{2}m_p l\dot{x}\dot{\theta}\cos\theta + \frac{1}{6}m_p l^2\dot{\theta}^2$$

Potential energy: (only exists for pole)
$$V = m_p g \frac{l}{2}\cos\theta$$

### Step 3: Construct Lagrangian

$$L = T - V = \frac{1}{2}(m_c + m_p)\dot{x}^2 + \frac{1}{2}m_p l\dot{x}\dot{\theta}\cos\theta + \frac{1}{6}m_p l^2\dot{\theta}^2 - m_p g \frac{l}{2}\cos\theta$$

### Step 5: Euler-Lagrange Equations

For coordinate $x$:
$$
\begin{align*}
F &= \frac{d}{dt}\frac{\partial L}{\partial \dot{x}} - \frac{\partial L}{\partial x} \\
\frac{\partial L}{\partial \dot{x}} &= (m_c + m_p)\dot{x} + \frac{1}{2}m_p l\dot{\theta}\cos\theta \\
\frac{d}{dt}\frac{\partial L}{\partial \dot{x}} &= (m_c + m_p)\ddot{x} + \frac{1}{2}m_p l\ddot{\theta}\cos\theta - \frac{1}{2}m_p l\dot{\theta}^2\sin\theta \\
\frac{\partial L}{\partial x} &= 0
\end{align*}
$$

Equation 1:
$$(m_c + m_p)\ddot{x} + \frac{1}{2}m_p l\ddot{\theta}\cos\theta - \frac{1}{2}m_p l\dot{\theta}^2\sin\theta = F$$

For coordinate $\theta$:
$$
\begin{align*}
F &= \frac{d}{dt}\frac{\partial L}{\partial \dot{\theta}} - \frac{\partial L}{\partial \theta} = 0\\
\frac{\partial L}{\partial \dot{\theta}} &= \frac{1}{2}m_p l\dot{x}\cos\theta + \frac{1}{3}m_p l^2\dot{\theta} \\
\frac{d}{dt}\frac{\partial L}{\partial \dot{\theta}} &= \frac{1}{2}m_p l\ddot{x}\cos\theta - \frac{1}{2}m_p l\dot{x}\dot{\theta}\sin\theta + \frac{1}{3}m_p l^2\ddot{\theta} \\
\frac{\partial L}{\partial \theta} &= -\frac{1}{2}m_p l\dot{x}\dot{\theta}\sin\theta + m_p g \frac{l}{2}\sin\theta
\end{align*}
$$

**Equation 2:**
$$\frac{1}{2}m_p l\ddot{x}\cos\theta + \frac{1}{3}m_p l^2\ddot{\theta} - m_p g \frac{l}{2}\sin\theta = 0$$

## Final System of Equations

$$\begin{align}
(m_c + m_p)\ddot{x} + \frac{1}{2}m_p l\ddot{\theta}\cos\theta - \frac{1}{2}m_p l\dot{\theta}^2\sin\theta &= F \\
\frac{1}{2}m_p l\ddot{x}\cos\theta + \frac{1}{3}m_p l^2\ddot{\theta} - m_p g \frac{l}{2}\sin\theta &= 0
\end{align}$$

## State Space Representation

Define state vector: $\mathbf{x} = [x, \dot{x}, \theta, \dot{\theta}]^T$

Solving for accelerations from the coupled equations above (Appendix A):

$$\ddot{x} = \frac{F + \frac{1}{2}m_p l\dot{\theta}^2\sin\theta - \frac{3}{2}m_p g\sin\theta\cos\theta}{m_c + m_p - \frac{3}{4}m_p\cos^2\theta}$$

$$\ddot{\theta} = \frac{-\frac{3}{2l}\cos\theta(F + \frac{1}{2}m_p l\dot{\theta}^2\sin\theta) + \frac{g}{2l}(m_c + m_p)\sin\theta}{m_c + m_p - \frac{3}{4}m_p\cos^2\theta}$$

## Linearization About Equilibrium

For small angles ($\sin\theta \approx \theta$, $\cos\theta \approx 1$, $\dot{\theta}^2 \approx 0$):

$$\begin{align}
\ddot{x} &\approx \frac{-\frac{3}{2}m_pg}{m_c + m_p - \frac{3}{4}m_p}\theta + \frac{1}{m_c + m_p - \frac{3}{4}m_p}F\\
&\approx \frac{-\frac{3}{2}m_pg}{m_c + \frac{1}{4}m_p}\theta + \frac{1}{m_c + \frac{1}{4}m_p}F\\
&\approx \frac{-6m_pg}{4m_c + m_p}\theta + \frac{4}{4m_c + m_p}F\\
\ddot{\theta} &\approx \frac{\frac{g}{2l}(m_c + m_p)}{m_c + \frac{1}{4}m_p}\theta - \frac{\frac{3}{2l}}{m_c + \frac{1}{4}m_p}F\\
&\approx \frac{2g(m_c + m_p)}{l(4m_c + m_p)}\theta - \frac{6}{l(4m_c + m_p)}F
\end{align}$$

## System Parameters

From the implementation:
- $m_c = 1.0$ kg (cart mass)
- $m_p = 0.1$ kg (pole mass)  
- $l = 1.0$ m (pole length)
- $g = 9.81$ m/s² (gravity)

This mathematical model forms the basis for the LQR controller implementation in the simulation.

## Appendix
### A. Solving for state space
As this seemed simple enough, I did it by hand. 
#### 1. Solve for $\ddot\theta$:
Equations 8 and 9 derived from equations 1 and 2 respectively.
$$
\begin{align}
\ddot{x}&=\frac{F}{m_c+m_p}+\frac{m_pl\dot{\theta}^2\sin\theta}{2(m_c+m_p)}-\frac{m_pl\cos\theta}{2(m_c+m_p)}\ddot{\theta}\\
\ddot{x}&=\frac{g\sin\theta}{\cos\theta}-\frac{2l\ddot\theta}{3\cos\theta}
\end{align}
$$
Using equations 8 and 9, solve for $\ddot\theta$
$$
\begin{align*}
\left(\frac{m_pl\cos\theta}{2(m_c+m_p)}-\frac{2l}{\cos\theta}\right)\ddot\theta&=\frac{F}{m_c+m_p}+\frac{m_pl\dot\theta^2\sin\theta}{2(m_c+m_p)}-\frac{g\sin\theta}{\cos\theta} \\
\frac{3m_pl\cos^2{\theta}-4l(m_c+m_p)}{6(m_c+m_p)\cos\theta}\ddot\theta&=\frac{2F\cos\theta+m_pl\dot\theta^2\sin\theta\cos\theta-2g(m_c+m_p)\sin\theta}{2(m_c+m_p)\cos\theta} \\
\ddot\theta&=\frac{6F\cos\theta+3m_pl\dot\theta^2\sin\theta\cos\theta-2g(m_c+m_p)\sin\theta}{3m_pl\cos^2\theta-4l(m_c+m_p)}
\end{align*}
$$
#### 2. Solve for $\ddot x$
Equations 10 and 11 derived from equations 1 and 2 respectively
$$
\begin{align}
\ddot\theta&=\frac{2F}{m_pl\cos\theta}-\frac{2(m_c+m_p)}{m_pl\cos\theta}\ddot x + \dot\theta^2\frac{\sin\theta}{\cos\theta}\\
\ddot\theta&=\frac{3g}{2l}\sin\theta-\frac{3}{2l}\cos\theta\ddot x
\end{align}
$$
Using equations 10 and 11, solve for $\ddot x$
$$
\begin{align*}
\left(\frac{2(m_c+m_p)}{m_pl\cos\theta}-\frac{3}{2l}\cos\theta\right)\ddot x &=\frac{2F}{m_pl\cos\theta}+\dot\theta^2\frac{\sin\theta}{\cos\theta}-\frac{3g\sin\theta}{2l} \\
\frac{4(m_c+m_p)-3m_p\cos^2\theta}{2m_pl\cos\theta}\ddot x&=\frac{4F+2\dot\theta^2m_pl\sin\theta-6gm_p\cos\theta\sin\theta}{2m_pl\cos\theta}\\
\ddot x&=\frac{4F+2\dot\theta^2m_pl\sin\theta-6m_pg\cos\theta\sin\theta}{4(m_c+m_p)-3m_p\cos^2\theta}
\end{align*}
$$