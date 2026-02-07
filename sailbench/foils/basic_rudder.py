"""Basic rudder foil model."""

import numpy as np

import sailbench.utils.coordinate_helper as utils
from sailbench.models.foil import Foil
from sailbench.models.model import State


class BasicRudder(Foil):
    """Basic rudder foil model."""

    def __init__(self, params: dict) -> None:
        """Initialize the BasicRudder model.

        Args:
            params (dict): Dictionary of parameters for the rudder model.

        """
        super().__init__(params)

    def compute(self, state: State, angle_input: float = 0.0) -> np.ndarray:
        """Compute the lift and drag coefficients for the rudder.

        Args:
            state (np.ndarray): Current boat state.
            angle_input (float): Angle of rudder in degrees.

        Returns:
            np.ndarray: Returns X and Y forces in newtons (within rudder frame)

        """
        # get local track vector
        global_track_vector = np.array([[state.u],
                                [state.v]])
        local_track_vector = utils.global_to_local(global_track_vector, state.psi)
        


        #Rotation matrix: local to rudder frame
        rotate_into_rudder = np.array([
        [np.cos(np.radians(angle_input)), -1 * np.sin(np.radians(angle_input))],
        [np.sin(np.radians(angle_input)), np.cos(np.radians(angle_input))],
        ])

        rudder_frame_track = rotate_into_rudder @ local_track_vector

        #angle of attack is angle of vector of track in rudder frame
        angle_of_track = np.arctan2(rudder_frame_track[1, 0] , rudder_frame_track[0, 0])
        
        # get lift and drag coefficients
        cl, cd = self.cl_cd(angle_of_track, re=self.p.get("re", 1e5))

        # compute dynamic pressure
        rho = self.p.get("water_density", 1000.0)  # kg/m^3
        u = state.u
        v = state.v
        v = np.hypot(u, v)
        q = 0.5 * rho * v**2

        # compute forces
        s = self.p.get("area", 1.0)  # m^2
    
        drag = cd * q * s
        lift = cl * q * s

        # return forces in fluid frame (X forward, Y starboard)
        fx = -drag
        fy = lift

        rotate_into_rudder = np.array([
            [np.cos(angle_of_track), np.sin(angle_of_track)],
        [-1* np.sin(angle_of_track), np.cos(angle_of_track)],
        ])
        # convert rudder frame to local frame
        rotate_into_local = np.array([
        [np.cos(np.radians(angle_input)), np.sin(np.radians(angle_input))],
        [-1* np.sin(np.radians(angle_input)), np.cos(np.radians(angle_input))],
        ])

        # rotate lift and drag fluid --> rudder --> local frame
        f_prime = (rotate_into_local) @ (rotate_into_rudder) @ np.array([[fx],[fy]])

        return np.asarray([f_prime[0,0], f_prime[1,0]])
