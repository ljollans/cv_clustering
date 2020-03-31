import numpy as np

def ecdf(x):
    xs = np.sort(x)
    ys = np.arange(1, len(xs) + 1) / float(len(xs))
    return xs, ys


def get_pac(con_mat):
    x1 = 0.1
    x2 = 0.9

    p = con_mat[np.tril_indices(con_mat.shape[0])].flatten()

    xs, ys = ecdf(p)
    select_vals = np.where(np.logical_and(xs >= x1, xs <= x2))
    x1_x2_range = ys[select_vals[0]]

    x1_val = x1_x2_range[0]
    x2_val = x1_x2_range[-1]
    pac = x2_val - x1_val

    return pac
