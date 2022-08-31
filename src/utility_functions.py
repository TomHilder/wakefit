import numpy as np

def find_max_list_index(my_list):
    lengths = [len(lst) for lst in my_list]
    return np.argmax(
        np.array(lengths)
    )

def find_max_list(my_list):
    ind = find_max_list_index(my_list)
    return my_list[ind]