from .read_M9           import Moment_9
from .utility_functions import find_max_list
from typing             import List
from decimal            import Decimal
import numpy                as np
import matplotlib.pyplot    as plt

class IsovelocityContoursM9():

    N_CONTOURS = 0
    V_MAX      = 0
    VELOCITIES = None
    CONTOURS   = None

    def __init__(
        self,
        M9:    Moment_9,
        dv:    float,
        v_max: float,
    ) -> None:
        """Create IsovelocityContoursM9 object from given Moment_9 object.

        M9:    Moment_9 object
        dv:    Channel spacing in km/s of cube from which M9 map was made
        v_max: Maximum velocity channel in km/s to find contours to. Must 
               be integer multiple of dv
        """

        # Check that v_max is integer multiple of dv
        if Decimal(f'{v_max}') % Decimal(f'{dv}') != Decimal('0.0'):
            raise Exception("v_max must be integer multiple of dv.")

        # calculate number of contours
        self.N_CONTOURS = int(2 * v_max / dv + 1)

        # get velocities
        self.V_MAX = v_max * 1e3    # convert to m/s
        self.VELOCITIES = np.linspace(
            -self.V_MAX, self.V_MAX, self.N_CONTOURS
        )
        
        # get matplotlib contours object
        xc      = M9.angular_coords[0]
        yc      = M9.angular_coords[1]
        d       = M9.data
        self.cs = plt.contour(xc, yc, d, levels=self.VELOCITIES)
        plt.close('all')

    def find_contours(self) -> List[np.ndarray]:

        # check if already calculated
        if self.CONTOURS is not None:
            return self.CONTOURS

        # get all polygon segments
        all_segs = self.cs.allsegs

        # keep only longest polygon segments (remove noisey crap)
        contours = []
        for level in all_segs:
            contours.append(find_max_list(level))

        self.CONTOURS = contours

        return contours
    
