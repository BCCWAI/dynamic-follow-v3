import numpy as np


def normX(data):
    v_ego_scale = [data.min(), data.max()]
    data_normalized = np.interp(data, v_ego_scale, [0, 1])
    return data_normalized, v_ego_scale
