import numpy as np

def rotation_matrix(ang, ax='x'):
    """Function for rotation matrices"""
    
    # get ang in [-2pi, 2pi]
    ang = ang % (2*np.pi)

    # get phi in [0, 2pi]
    if ang < 0:
        ang = 2*np.pi + ang
    
    if ax == "x":
        return [
            [1,           0,            0],
            [0, np.cos(ang), -np.sin(ang)],
            [0, np.sin(ang),  np.cos(ang)]
        ]
    elif ax == "y":
        return [
            [ np.cos(ang), 0, np.sin(ang)],
            [ 0,           1,           0],
            [-np.sin(ang), 0, np.cos(ang)]
        ]
    elif ax == "z":
        return [
            [np.cos(ang), -np.sin(ang), 0],
            [np.sin(ang),  np.cos(ang), 0],
            [0,                      0, 1]
        ]
    else:
        raise ValueError("ax must be x, y or z")