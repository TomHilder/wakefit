import numpy as np

def discretise_data(data, bins):

    N_1 = data.shape[0]
    N_2 = data.shape[1]

    discrete_data = np.zeros(data.shape)

    for i in range(N_1):
        for j in range(N_2):
            if np.isnan(data[i,j]):
                discrete_data[i,j] = np.nan
            elif data[i,j] <= bins[0] or data[i,j] > bins[-1]:
                discrete_data[i,j] = np.nan
            else:
                for k in range(len(bins)-1):
                    if data[i,j] > bins[k] and data[i,j] <= bins[k+1]:
                        discrete_data[i,j] = bins[k]

    return discrete_data