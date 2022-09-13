from dis import dis
import numpy                as np
import matplotlib.pyplot    as plt
from scipy.interpolate  import griddata
from src.discretise     import discretise_data
from src.model          import DiskModel
from src.point_dist     import calc_pp_dist_arr_chi_sq

def find_max_list_index(my_list):
    lengths = [len(lst) for lst in my_list]
    try:
        return np.argmax(
            np.array(lengths)
        )
    except:
        raise Exception

def find_max_list(my_list):
    try:
        ind = find_max_list_index(my_list)
        return my_list[ind]
    except Exception:
        raise Exception

def find_contours(cs):

    # get all polygon segments
    all_segs = cs.allsegs

    contours = []
    for level in all_segs:
        try:
            contours.append(find_max_list(level))
        except Exception:
            contours.append(np.array([[0,0],[0,0]]))

    return contours

def get_chi_sq(
    x,
    p,
    a_cw,
    r_outer,
    distance,
    pos_angle,
    planet_az,
    inclination,
    M9_map,
    dM9_map,
    discrete_data,
    contours_data,
    velocity_channels
):

    m_star = x[0]
    hr = np.abs(x[1])
    psi = x[2]
    q = x[3]
    z0 = x[4]
    r_taper = x[5]
    q_taper = x[6]

    print(x)

    dd = discrete_data

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

    try:
        interpolated_v = griddata(points, mvv, grid, method="linear", fill_value=np.nan)
    except:
        return 1e10

    csm = plt.contour(oxx, oyy, interpolated_v, cmap="RdBu_r", levels=velocity_channels)
    contours_model = find_contours(csm)
    plt.close('all')

    distance = 0.0
    for i in range(len(contours_data)):
        points1 = contours_data[i]
        points2 = contours_model[i]
        temp_dist = calc_pp_dist_arr_chi_sq(points1, points2, 0.15)
        distance += temp_dist

    n_data_contours = 13948
    red_chi_sq_cont = distance / (n_data_contours - 13)

    print(red_chi_sq_cont)

    #residuals_not_discrete = M9_map.data - interpolated_v

    #chi_sq_not_discrete = np.nansum((M9_map.data - interpolated_v)**2 / dM9_map.data**2)
    #red_chi_sq_not_discrete = chi_sq_not_discrete / (2048**2 - 13)

    #dm = discretise_data(interpolated_v, velocity_channels)

    #residuals = dd - dm

    #chi_sq = np.nansum((dd - dm)**2 / dM9_map.data**2)
    #red_chi_sq = chi_sq / (2048**2 - 13)

    #return (red_chi_sq, red_chi_sq_cont, red_chi_sq_not_discrete), (residuals, residuals_not_discrete)
    return red_chi_sq_cont