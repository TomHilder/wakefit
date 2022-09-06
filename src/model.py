from pickle import FALSE
import numpy        as np
from dis        import dis
from astropy    import constants as c
from .rotations import rotation_matrix

# all constants given in CGS units
m_solar    = c.M_sun.si.value
m_jupiter  = c.M_jup.si.value
G_const    = c.G.si.value
au         = c.au.si.value

class DiskModel():

    DISTANCE = 0.0

    rotated = False

    def __init__(
        self,
        distance,   # don't fit ?
        r_outer,    # don't fit
        a_cw,       # don't fit
        m_star,
        hr,
        psi,
        p,
        q,
        z0,
        r_taper,
        q_taper,
        grid = "cartesian"
        ) -> None:

        self.DISTANCE = distance

        r0  = 1

        n_r   = 100
        n_phi = 160

        n_x = 200
        n_y = 201
        
        # setup grid coordinates
        if grid == "cartesian":
            self.x, self.y   = self.setup_grid_cart(n_x, n_y, r_outer)
            self.r, self.phi = self.get_polar(self.x, self.y)
        elif grid == "polar":
            self.r, self.phi = self.setup_grid_polar(n_r, n_phi, r_outer, r_inner=0.5)
            self.x, self.y   = self.get_cartesian(self.r, self.phi)
        else:
            raise Exception('choose grid="cartesian" or "polar"')

        # find disc height
        self.z = self.get_height(self.r, q, z0, r0, r_taper, q_taper)

        # azimuthal velocities with pressure and height in m/s
        self.v_phi = self.get_velocities(self.r, self.z, m_star, a_cw, hr, r0, psi, p)

        # get cartesian velocities
        v_x = -self.v_phi * np.sin(self.phi) # + self.v_r * np.cos(self.phi)
        v_y =  self.v_phi * np.cos(self.phi) # + self.v_r * np.sin(self.phi)
        v_z = np.zeros(v_x.shape)

        # define cartesian velocity field
        self.v_field = np.array([v_x, v_y, v_z])

    def setup_grid_polar(
        self,
        n_r,
        n_phi,
        r_outer,
        r_inner = 0.0
    ):

        r   = np.linspace(r_inner, r_outer, n_r)
        phi = np.linspace(0.0, 2*np.pi, n_phi)

        return np.meshgrid(r, phi)

    def setup_grid_cart(
        self,
        n_x,
        n_y,
        r_outer
    ):

        x = np.linspace(-r_outer, r_outer, n_x)
        y = np.linspace(-r_outer, r_outer, n_y)

        return np.meshgrid(x, y)
 
    def get_cartesian(
        self,
        r,
        phi
    ):
        x = r * np.cos(phi)
        y = r * np.sin(phi)

        return x, y

    def get_polar(
        self,
        x,
        y
    ):

        r   = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)

        return r, phi

    def get_sky_coords(self) -> None:

        self.x_sky = self.x / self.DISTANCE
        self.y_sky = self.y / self.DISTANCE
        self.z_sky = self.z / self.DISTANCE
        
    def get_height(
        self,
        r:          np.ndarray, 
        q:          float,
        z0:         float, 
        r0:         float = 1.0,
        r_taper:    float = np.inf,
        q_taper:    float = 1.0
    ) -> np.ndarray:

        rr = r / self.DISTANCE

        return z0 * ( (rr / r0)**q ) * np.exp( -(rr / r_taper)**q_taper ) * self.DISTANCE

    def get_velocities(
        self, 
        r,
        z,
        m_star, 
        a_cw, 
        hr, 
        r0, 
        psi, 
        p
    ) -> np.ndarray:

        v_kep = np.sqrt(G_const * m_star * m_solar / (r * au))
        v_phi = a_cw * v_kep

        hr_func = hr * (r / r0) ** (0.5 - psi)

        # height and pressure terms correction
        corr = np.sqrt(
            (-1* (p + 2 * psi) * hr_func**2) + (1 - 2 * psi) + (2 * psi * r / np.sqrt(r**2 + z**2))
        )

        return v_phi * corr
    
    def rotate(
        self,
        pos_ang     = 0.,
        inc         = 0.,
        planet_az   = 0.,
        grid_rotate = True
    ):
        if self.rotated:
            return

        print("edge height = ", self.z[0,0])
        print("max height = ", np.max(self.z))

        # get number of points
        N_X = self.x.shape[1]
        N_Y = self.x.shape[0]
        assert self.y.shape[1] == N_X
        assert self.y.shape[0] == N_Y

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
        for i in range(N_X):
            for j in range(N_X):

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