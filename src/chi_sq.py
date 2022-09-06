from dis import dis
import numpy                as np
from scipy.interpolate  import griddata
from src.read_M9        import Moment_9
from src.discretise     import discretise_data
from src.model          import DiskModel

def get_chi_sq(
    distance,
    r_outer,
    a_cw,
    m_star,
    hr,
    psi,
    p,
    q,
    z0,
    r_taper,
    q_taper,
    pos_angle,
    planet_az,
    inclination
):

    M9_file  = "peak_velocity/HD_163296_CO_220GHz.0.15arcsec.JvMcorr.image_v0.fits"
    dM9_file = "peak_velocity/HD_163296_CO_220GHz.0.15arcsec.JvMcorr.image_dv0.fits"

    M9_map = Moment_9(M9_file, v_system=5.76)
    dM9_map = Moment_9(dM9_file)

    max_vel_channel = 2.6*1e3
    n_vels  = int(2*2.6/0.2 + 1)
    velocity_channels = np.linspace(-max_vel_channel, max_vel_channel, n_vels)

    dd = discretise_data(M9_map.data, velocity_channels)

    model = DiskModel(
        distance=distance,
        r_outer=r_outer, 
        a_cw=a_cw, 
        m_star=m_star, 
        hr=hr, 
        psi=psi, 
        p=p, 
        q=q,
        z0=z0,
        r_taper=r_taper,
        q_taper=q_taper
    )

    model.rotate(pos_ang=pos_angle, inc=inclination, planet_az=planet_az)
    mx = model.rot_x / model.DISTANCE
    my = model.rot_y / model.DISTANCE
    mv = model.rot_v_field[2,:,:]

    # mask the stuff in the model beyond r_outer
    nan_array = np.zeros(model.x.shape)
    nan_array[:] = np.nan
    ddd = np.sqrt(model.x**2 + model.y**2)
    vv = np.where(ddd < r_outer, mv, nan_array)

    ox, oy = M9_map.angular_coords

    # create masked array
    mask = np.isnan(vv)
    m_vv = np.ma.array(data=vv, mask=mask)

    # remove points based on mask
    mxx = mx[~m_vv.mask]
    myy = my[~m_vv.mask]
    mvv = vv[~m_vv.mask]

    points = (mxx, myy)

    oxx, oyy = np.meshgrid(ox, oy)
    grid = (oxx, oyy)

    interpolated_v = griddata(points, mvv, grid, method="linear", fill_value=np.nan)

    residuals_not_discrete = M9_map.data - interpolated_v

    chi_sq_not_discrete = np.nansum((M9_map.data - interpolated_v)**2 / dM9_map.data**2)
    red_chi_sq_not_discrete = chi_sq_not_discrete / (2048**2 - 13)

    dm = discretise_data(interpolated_v, velocity_channels)

    residuals = dd - dm

    chi_sq = np.nansum((dd - dm)**2 / dM9_map.data**2)
    red_chi_sq = chi_sq / (2048**2 - 13)

    return (red_chi_sq, red_chi_sq_not_discrete), (residuals, residuals_not_discrete)