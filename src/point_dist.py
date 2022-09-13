import numpy as np

def calc_pp_dist(set_points1, set_points2):
    """give two lists of points aka two lists of tuples (x,y)"""

    n_set1 = len(set_points1)
    n_set2 = len(set_points2)

    arr_1 = np.array(set_points1)
    arr_2 = np.array(set_points2)

    print(arr_1.shape)
    
    total = 0.0

    # loop over all points in set 1
    for i in range(n_set1):

        distances = np.linalg.norm(arr_1[i,:] - arr_2, axis=1)
        min_index = np.argmin(distances)

        total += distances[min_index]

    return total

def calc_pp_dist_arr(set_points1, set_points2):
    """give two lists of points in 2D arrays of shape(N, 2)"""

    n_set1 = set_points1.shape[0]
    n_set2 = set_points2.shape[0]

    arr_1 = set_points1
    arr_2 = set_points2
    
    total = 0.0

    # loop over all points in set 1
    for i in range(n_set1):

        distances = np.linalg.norm(arr_1[i,:] - arr_2, axis=1)
        min_index = np.argmin(distances)

        total += distances[min_index]

    return total

def calc_pp_dist_arr_chi_sq(set_points1, set_points2, u_dist):
    """give two lists of points in 2D arrays of shape(N, 2)"""

    n_set1 = set_points1.shape[0]
    n_set2 = set_points2.shape[0]

    arr_1 = set_points1
    arr_2 = set_points2
    
    total = 0.0

    # loop over all points in set 1
    for i in range(n_set1):

        distances = np.linalg.norm(arr_1[i,:] - arr_2, axis=1)
        min_index = np.argmin(distances)

        total += (distances[min_index]**2 / u_dist**2)

    return total