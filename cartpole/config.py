"""Configuration classes for cartpole simulation."""

from dataclasses import dataclass
import numpy as np


class CartPolePhysicsConfig:
    def __init__(self, cart_mass, pole_mass, pole_length, pole_diam):

        if cart_mass is None: raise Exception("Cart Mass cannot be None")
        if pole_mass is None: raise Exception("Pole Mass cannot be None")
        if pole_length is None: raise Exception("Pole length cannot be None")

        print(f"Cart Mass: {cart_mass}\nPole Mass: {pole_mass}\nPole Length: {pole_length}")
        
        # Cart parameters
        self.cart_mass: float = cart_mass

        # Pole parameters
        self.pole_length: float = pole_length
        self.pole_mass: float = pole_mass
        self.pole_diam: float = pole_diam

        # Constants
        self.gravity: float = 9.81
    

    def compute_system_matrices(self, pole_geom='prism') -> tuple[np.ndarray, np.ndarray]:
        if pole_geom=='prism':
            I = self.pole_mass*(4*self.pole_length**2+self.pole_diam**2)
        elif pole_geom=='cylinder':
            I = self.pole_mass*(4*self.pole_length**2+3*(self.pole_diam/2)**2)/12

        pm = self.pole_mass*self.pole_length/2
        M = self.cart_mass + self.pole_mass
        delta = I*M-pm**2

        g = self.gravity

        # Linearized system matrices from proper derivation
        A = np.array([
            [0, 1, 0, 0],
            [0, 0, -(g*pm**2)/delta, 0],
            [0, 0, 0, 1],
            [0, 0, (pm*g*M)/delta, 0]
        ])

        B = np.array([
            [0],
            [I/delta],
            [0],
            [-pm/delta],
        ])

        return A, B

