from pickle import FALSE
import numpy                        as np
from astropy                    import constants
from wakeflow.wakefit_interface import _WakefitModel
from .rotations                 import rotation_matrix
from typing                     import Tuple

# all constants given in CGS units
M_SOL  = constants.M_sun.si.value
M_JUP  = constants.M_jup.si.value
G_CONS = constants.G.si.value
AU     = constants.au.si.value

class DiskModel():

    # grid parameters
    N_X = 0
    N_Y = 0

    # radii in AU
    R_OUTER  = 0.0      # disk outer radius
    R_TAPER  = 0.0      # critical radius for exp. taper on height
    R_REF    = 0.0      # radius where HR_REF and Z_REF are taken
    R_PLANET = 0.0      # orbital radius of the planet

    # power law indices
    CS_IND   = 0.0      # sound speed
    P_IND    = 0.0      # density
    Q_IND    = 0.0      # height
    Q_TAPER  = 0.0      # taper on height

    # reference height in AU
    Z_REF = 0.0

    # disk aspect ratio
    HR_REF = 0.0

    # rotation direction (1 is anticlockwise)
    A_CW = 1

    # central star mass in M_solar
    M_STAR = 0.0

    # mass of planet in m_Jupiter
    M_PLANET = 0.0

    # has the model been rotated?
    rotated = False

    def __init__(
        self,
        r_outer,
        a_cw,
        m_star,
        m_planet,
        r_planet,
        hr,
        psi,
        p,
        q,
        z0,
        r_taper,
        q_taper,
        r_ref,
        n_x = 200,
        n_y = 201,
        ) -> None:

        # setup parameters as constants for model
        self.N_X      = n_x
        self.N_Y      = n_y
        self.R_OUTER  = r_outer
        self.R_TAPER  = r_taper
        self.R_REF    = r_ref
        self.R_PLANET = r_planet
        self.CS_IND   = psi
        self.P_IND    = p
        self.Q_IND    = q
        self.Q_TAPER  = q_taper
        self.Z_REF    = z0
        self.HR_REF   = hr
        self.A_CW     = a_cw
        self.M_STAR   = m_star
        self.M_PLANET = m_planet
        
        # setup grid coordinates
        self.x, self.y   = self.setup_grid_cart()
        self.r, self.phi = self.get_polar()

        # find disc height
        self.z = self.get_height()

        # azimuthal velocities with pressure and height in m/s
        self.v_phi = self.get_rotation_velocity()

        # radial velocities initialised to zero
        self.v_r  = np.zeros(self.v_phi.shape)

        # calcualte velocity perturbations from the planet
        wake = self.get_planet_wake()
        self.wake_v_r   = np.transpose(wake[2][:,0,:]) * 1.e3 # convert to SI units from km/s 
        self.wake_v_phi = np.transpose(wake[3][:,0,:]) * 1.e3 # convert to SI units from km/s 

        # add velocity perturbations to background
        self.v_phi += self.wake_v_phi
        self.v_r   += self.wake_v_r

        # get cartesian velocities
        v_x = -self.v_phi * np.sin(self.phi) + self.v_r * np.cos(self.phi)
        v_y =  self.v_phi * np.cos(self.phi) + self.v_r * np.sin(self.phi)
        v_z = np.zeros(v_x.shape)

        # define cartesian velocity field
        self.v_field = np.array([v_x, v_y, v_z])

    def setup_grid_cart(self) -> np.ndarray:

        # get grid coordinates
        x = np.linspace(-self.R_OUTER, self.R_OUTER, self.N_X)
        y = np.linspace(-self.R_OUTER, self.R_OUTER, self.N_Y)

        # return meshgrids of coords
        return np.meshgrid(x, y)

    def get_polar(self) -> Tuple[np.ndarray]:

        # perform transformations
        r   = np.sqrt(self.x**2 + self.y**2)
        phi = np.arctan2(self.y, self.x)

        # return transformed coords
        return r, phi
        
    def get_height(self) -> np.ndarray:

        # calculate and return height using model constants
        return self.Z_REF * ( (self.r / self.R_REF)**self.Q_IND ) * np.exp( -(self.r / self.R_TAPER)**self.Q_TAPER )

    def get_rotation_velocity(self) -> np.ndarray:

        v_kep = np.sqrt(G_CONS * self.M_STAR * M_SOL / (self.r * AU))
        v_phi = self.A_CW * v_kep

        hr_func = self.HR_REF * (self.r / self.R_REF) ** (0.5 - self.CS_IND)

        # height and pressure terms correction
        corr = np.sqrt(
            (-1* (self.P_IND + 2 * self.CS_IND) * hr_func**2) + \
            (1 - 2 * self.CS_IND) + \
            (2 * self.CS_IND * self.r / np.sqrt(self.r**2 + self.z**2))
        )

        return v_phi * corr

    def get_planet_wake(self) -> Tuple[np.ndarray]:

        # initialise wakeflow model
        wake_model = _WakefitModel()

        if self.A_CW == 1:
            cw_rot = False
        elif self.A_CW == -1:
            cw_rot = True
        else:
            raise Exception("Rotation direction undefined.")

        # configure model appropriately
        wake_model.configure(
            m_star      = self.M_STAR,
            m_planet    = self.M_PLANET,
            r_outer     = self.R_OUTER,
            r_planet    = self.R_PLANET,
            r_ref       = self.R_REF,
            q           = self.CS_IND,
            p           = self.P_IND,
            hr          = self.HR_REF,
            cw_rotation = cw_rot,
            grid_type   = "cartesian",
            n_x         = self.N_X,
            n_y         = self.N_Y,
            n_z         = 1,
            make_midplane_plots = False,
            show_midplane_plots = False,
            save_perturbations  = False,
            save_total          = False
        )

        # generate and return perturbations
        return wake_model.run()

    def rotate(
        self,
        pos_ang     = 0.0,
        inc         = 0.0,
        planet_az   = 0.0,
        grid_rotate = True  # rarely would ever want to change this
    ):
        if self.rotated:
            return

        #print("edge height = ", self.z[0,0])
        #print("max height = ", np.max(self.z))

        assert self.y.shape[1] == self.N_X
        assert self.y.shape[0] == self.N_Y

        # convert to radians
        pos_ang     *= np.pi / 180.
        planet_az   *= np.pi / 180.
        inc         *= np.pi / 180.

        # rotation matrices
        rot_pl_z = rotation_matrix(-planet_az, "z")
        rot_in_x = rotation_matrix(       inc, "x")
        rot_pa_z = rotation_matrix(   pos_ang, "z")

        # copy arrays
        X = np.copy(self.x)
        Y = np.copy(self.y)
        Z = np.copy(self.z)
        V = np.copy(self.v_field)

        # loop over all points
        for i in range(self.N_X):
            for j in range(self.N_X):

                # rotate grid
                if grid_rotate:

                    X[i,j], Y[i,j], Z[i,j] = np.dot(rot_pl_z, [X[i,j], Y[i,j], Z[i,j]])
                    X[i,j], Y[i,j], Z[i,j] = np.dot(rot_in_x, [X[i,j], Y[i,j], Z[i,j]])
                    X[i,j], Y[i,j], Z[i,j] = np.dot(rot_pa_z, [X[i,j], Y[i,j], Z[i,j]])

                # rotate around the normal axis of the disk, corresponding the planet_az angle
                V[:,i,j] = np.dot(rot_pl_z, V[:,i,j])

                # rotate around the x-axis of the sky plane to match the inclination
                V[:,i,j] = np.dot(rot_in_x, V[:,i,j])

                # rotate around the normal axis of the sky plane to match the PA
                V[:,i,j] = np.dot(rot_pa_z, V[:,i,j])

        # set rotated arrays
        self.rot_x       = X
        self.rot_y       = Y
        self.rot_z       = Z
        self.rot_v_field = V

        self.rotated = True