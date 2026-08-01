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
...