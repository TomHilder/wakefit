import numpy        as np
from astropy.io import fits

class Moment_9():

    # constants
    V_SYSTEM = 0.0   # systemic velocity in m/s
    DISTANCE = None  # distance in parsecs

    # coordinates
    angular_coords  = None
    physical_coords = None

    # data
    data = None

    def __init__(
        self,
        file_loc:   str,
        v_system: float = None,
        distance: float = None
    ) -> None:
        """Create Moment_9 object from provided fits file containing M9 map.

        file_loc: path to file
        v_system: systematic velocity in km/s
        distance: distance in pc
        """

        # set system constants
        if v_system is not None: self.V_SYSTEM = v_system * 1e3 # convert to m/s
        if distance is not None: self.DISTANCE = distance
        
        # read data and header
        with fits.open(file_loc) as hdul:
            header = hdul[0].header
            data   = hdul[0].data

        # subtract systemic velocity
        self.data = data - self.V_SYSTEM

        # header info to get grid
        naxis1 = header['NAXIS1']
        naxis2 = header['NAXIS2']
        cdelt1 = header['CDELT1']
        cdelt2 = header['CDELT2']
        crpix1 = header['CRPIX1']
        crpix2 = header['CRPIX2']

        # get coordinate midpoints
        midpoint_ra  = crpix1 * cdelt1 * 3600
        midpoint_dec = crpix2 * cdelt2 * 3600

        # create angular coordinates grid
        X_ang = -np.linspace(-midpoint_ra,   midpoint_ra, naxis1)
        Y_ang =  np.linspace(-midpoint_dec, midpoint_dec, naxis2)

        # save angular coords
        self.angular_coords = (X_ang, Y_ang)

        # create physical coordinates if distance provided
        if self.DISTANCE is not None:

            # create grid
            X_phys = np.copy(X_ang) * self.DISTANCE
            Y_phys = np.copy(Y_ang) * self.DISTANCE

            # save coords
            self.physical_coords = (X_phys, Y_phys)
