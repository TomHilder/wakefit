import numpy as np

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