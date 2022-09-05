import numpy    as np
import scipy    as sp
import astropy  as ap

from astropy.io import fits

class CO_cube():
    
    DISTANCE    = 0
    ang_coords  = None

    def __init__(self, filename: str, distance: float = None) -> None:
        """Create CO cube object from provided fits file.
        filename: name of file
        v_syst:   systematic velocity in km/s
        distance: distance in pc
        """
        
        # open file
        file = f"{filename}"
        moment_1_fits = fits.open(file)

        # read data
        data = moment_1_fits[0].data

        # set constants
        if distance is not None: self.DISTANCE = distance

        self.data = data

        # get header info to get grid
        header = moment_1_fits[0].header
        naxis1 = header['NAXIS1']
        naxis2 = header['NAXIS2']
        cdelt1 = header['CDELT1']
        cdelt2 = header['CDELT2']
        crpix1 = header['CRPIX1']
        crpix2 = header['CRPIX2']

        # get coordinates
        midpoint_ra     = crpix1 * cdelt1 * 3600
        midpoint_dec    = crpix2 * cdelt2 * 3600

        # create grid in angular coordinates
        self.X_ang = -np.linspace(-midpoint_ra, midpoint_ra, naxis1)
        self.Y_ang = np.linspace(-midpoint_dec, midpoint_dec, naxis2)

        # create grid in physical coordinates
        self.X_phys = np.copy(self.X_ang) * self.DISTANCE
        self.Y_phys = np.copy(self.X_ang) * self.DISTANCE