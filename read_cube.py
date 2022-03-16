import numpy    as np
import scipy    as sp
import astropy  as ap


class CO_cube():

    V_SYST      = 0
    DISTANCE    = 0
    ang_coords  = None

    def __init__(self, filename: str, v_syst: float = None, distance: float = None) -> None:
        """Create CO cube object from provided fits file.
        filename: name of file
        v_syst:   systematic velocity in km/s
        distance: distance in pc
        """
        
        # open file
        file = f"observations/{filename}"
        moment_1_fits = ap.io.fits.open(file)

        # read data
        data = moment_1_fits[0].data

        # set constants
        if v_syst   is not None: self.V_SYST   = v_syst 
        if distance is not None: self.DISTANCE = distance

        # subtract systemic velocity
        self.data = data - v_syst

        # get header info to get grid
        header = moment_1_fits[0].header
        naxis1 = header['NAXIS1']
        naxis2 = header['NAXIS2']
        cdelt1 = header['CDELT1']
        cdelt2 = header['CDELT2']
        crpix1 = header['CRPIX1']
        crpix2 = header['CRPIX2']

        # get midpoint coordinates
        midpoint_ra     = crpix1 * cdelt1 * 3600
        midpoint_dec    = crpix2 * cdelt2 * 3600

        # create grid in angular coordinates
        self.X_ang = np.linspace(-midpoint_ra, midpoint_ra, naxis1)
        self.Y_ang = np.linspace(-midpoint_dec, midpoint_dec, naxis2)

        # create grid in physical coordinates
        self.X_phys = np.copy(self.X_ang) * distance
        self.Y_phys = np.copy(self.X_ang) * distance