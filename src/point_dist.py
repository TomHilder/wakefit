import numpy as np

def calc_pp_dist(set_points1, set_points2):
    """give two lists of points aka two lists of tuples (x,y)"""

    n_set1 = len(set_points1)
    n_set2 = len(set_points2)

    arr_1 = np.array(set_points1)
    arr_2 = np.array(set_points2)
    
    total = 0.0

    # loop over all points in set 1
    for i in range(n_set1):

        distances = np.linalg.norm(arr_1[i,:] - arr_2, axis=1)
        min_index = np.argmin(distances)

        total += distances[min_index]

    return total

