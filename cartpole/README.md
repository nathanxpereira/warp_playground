## Problem Statement
We are trying to solve the "Cartpole" problem. A pole is positioned on a cart and the cart moves side to side to balance the pole vertically. 

We are importing the NVIDIA built in model for the cartpole, which is a rectangle for the cart body, and a rectangnle as the pole. 

Let $x$ be the state of the system, and $\mu$ be a scalar horrizontal force applied to the cart. 

See the appendix for the system derivation. 

*This is designed for a cylindrical pole. Needs to be recalculated.* 
$$
\dot{x}=Ax+B\mu
$$
where,
$$
x=\left[ \begin{array}{}
x \\
\dot{x} \\
\theta \\
\dot{\theta}
\end{array} \right],\text{ }
A = \left[ \begin{array}{}
0 & 1 & 0 & 0 \\
0 & 0 & \frac{-3gm}{4M+m} & 0 \\
0 & 0 & 0 & 1 \\
0 & 0 & \frac{6(M+m)g}{(4M+m)l} & 0
\end{array} \right], \text{ } B = \left[ \begin{array}{}
0 \\
\frac{4}{4M+m} \\
0 \\
-\frac{6}{(4M+m)l}
\end{array}\right]
$$

### Defining the LQR problem


## Solving

## Appendix 
System derivation
$$
\begin{align*}
\mathcal{L} &= T - V \\
T &= T_{cart} + T_{pend}
\end{align*}
$$
Find Lagrangian
$$
\begin{align*}
T_{cart} &= \frac{1}{2}M\dot{x}^2 \\
T_{pend} &= \frac{1}{2}\left[\left(\dot{x}^2+\frac{l}{2}\dot{\theta}\cos\theta\right)^2+\left(-\frac{l}{2}\dot\theta\sin\theta\right)^2 \right]+\frac{1}{2}I_p\dot\theta^2 \\
T&=\frac{1}{2}(M+m)\dot{x}^2+\frac{ml}{2}\dot{x}\dot\theta\cos\theta + \frac{1}{2}I_p\dot\theta^2\\
V&=mg(\frac{l}{2}\cos\theta+y_0)
\end{align*}
$$
Formula for Lagrangian 1
$$
\begin{align*}
F&=\frac{\partial}{\partial{t}}\frac{\partial{\mathcal{L}}}{\partial{\dot{x}}}-\frac{\partial{\mathcal{L}}}{\partial{x}}\\
F&=(M+m)\ddot{x}+\frac{ml}{2}\ddot\theta\cos\theta-\frac{ml}{2}\dot\theta^2\sin\theta
\end{align*}
$$
Formula for Lagrangian 2
$$
\begin{align*}
0&=\frac{\partial}{\partial{t}}\frac{\partial{\mathcal{L}}}{\partial{\dot{\theta}}}-\frac{\partial{\mathcal{L}}}{\partial{\theta}}\\
0&=\frac{ml}{2}\ddot{x}\cos\theta+I_p\ddot\theta-\frac{mgl}{2}\sin\theta
\end{align*}
$$
Solve for $\ddot{x}$ and $\ddot\theta$
$$
\begin{align*}

\ddot{x}&=\frac{-\left(\frac{ml}{2}\right)^2g\sin\theta\cos\theta+\frac{ml}{2}I_p\dot\theta^2\sin\theta+I_pF}{I_p(M+m)-\left(\frac{ml}{2}\right)^2\cos^2\theta}\\
\ddot\theta&=\frac{\frac{ml}{2}g(M+m)\sin\theta-\left(\frac{ml}{2}\right)^2 \dot\theta^2\sin\theta\cos\theta-\frac{ml}{2}F\cos\theta}{I_p(M+m)-\left(\frac{ml}{2}\right)^2\cos^2\theta}
\end{align*}
$$
Linearize
$$
\begin{align*}
\Delta &= I_p(M+m)-\left(\frac{ml}{2}\right)^2\\
\ddot{x} &= \frac{-\left(\frac{ml}{2}\right)^2g}{\Delta}\theta+\frac{I_p}{\Delta}F \\
\ddot\theta &= \frac{\frac{ml}{2}g(M+m)}{\Delta}\theta-\frac{\frac{ml}{2}}{\Delta}F
\end{align*}
$$