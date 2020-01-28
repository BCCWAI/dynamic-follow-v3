import numpy as np
import functools
import operator
scale = [0.0, 1.0] # wider scale might improve accuracy

def interp_fast(x, min_max):
    return (x - min_max[0]) / (min_max[1] - min_max[0])

#tracks_normalized, car_data_normalized, scales = normTracks([[{'aRel': 0.0, 'vRel': -2.3499999046325684, 'oncoming': False, 'stationary': False, 'trackID': 4164, 'dRel': 62.79999923706055, 'yRel': 0.07999999821186066, 'status': 0.0}, {'aRel': 0.0, 'vRel': -2.1500000953674316, 'oncoming': False, 'stationary': False, 'trackID': 4165, 'dRel': 64.12000274658203, 'yRel': -0.03999999910593033, 'status': 0.0}, {'aRel': 0.0, 'vRel': 2.549999952316284, 'oncoming': False, 'stationary': False, 'trackID': 4166, 'dRel': 184.0399932861328, 'yRel': 3.0799999237060547, 'status': 0.0}, {'aRel': 0.0, 'vRel': -1.875, 'oncoming': False, 'stationary': False, 'trackID': 4109, 'dRel': 58.2400016784668, 'yRel': -3.0399999618530273, 'status': 0.0}, {'aRel': 0.0, 'vRel': -1.899999976158142, 'oncoming': False, 'stationary': False, 'trackID': 4134, 'dRel': 96.72000122070312, 'yRel': -3.359999895095825, 'status': 0.0}, {'aRel': 0.0, 'vRel': 5.875, 'oncoming': False, 'stationary': False, 'trackID': 4013, 'dRel': 66.31999969482422, 'yRel': 3.319999933242798, 'status': 0.0}, {'aRel': 0.0, 'vRel': -1.4500000476837158, 'oncoming': False, 'stationary': False, 'trackID': 4146, 'dRel': 35.20000076293945, 'yRel': -3.5199999809265137, 'status': 0.0}, {'aRel': 0.0, 'vRel': -1.5, 'oncoming': False, 'stationary': False, 'trackID': 4085, 'dRel': 35.20000076293945, 'yRel': -3.5999999046325684, 'status': 0.0}, {'aRel': 0.0, 'vRel': -2.200000047683716, 'oncoming': False, 'stationary': False, 'trackID': 4150, 'dRel': 64.04000091552734, 'yRel': -0.0, 'status': 0.0}, {'aRel': 0.0, 'vRel': -1.5, 'oncoming': False, 'stationary': False, 'trackID': 4151, 'dRel': 81.87999725341797, 'yRel': -0.0, 'status': 0.0}, {'aRel': 0.0, 'vRel': -1.975000023841858, 'oncoming': False, 'stationary': False, 'trackID': 4156, 'dRel': 57.79999923706055, 'yRel': -2.680000066757202, 'status': 0.0}], [{'aRel': 0.0, 'vRel': -2.4000000953674316, 'oncoming': False, 'stationary': False, 'trackID': 4164, 'dRel': 62.720001220703125, 'yRel': 0.07999999821186066, 'status': 0.0}, {'aRel': 0.0, 'vRel': -2.0999999046325684, 'oncoming': False, 'stationary': False, 'trackID': 4165, 'dRel': 63.68000030517578, 'yRel': -0.0, 'status': 0.0}, {'aRel': 0.0, 'vRel': 2.6500000953674316, 'oncoming': False, 'stationary': False, 'trackID': 4166, 'dRel': 184.47999572753906, 'yRel': 3.0799999237060547, 'status': 0.0}, {'aRel': 0.0, 'vRel': -1.850000023841858, 'oncoming': False, 'stationary': False, 'trackID': 4109, 'dRel': 57.560001373291016, 'yRel': -3.1600000858306885, 'status': 0.0}, {'aRel': 0.0, 'vRel': -1.899999976158142, 'oncoming': False, 'stationary': False, 'trackID': 4134, 'dRel': 96.4000015258789, 'yRel': -3.359999895095825, 'status': 0.0}, {'aRel': 0.0, 'vRel': 5.900000095367432, 'oncoming': False, 'stationary': False, 'trackID': 4013, 'dRel': 67.23999786376953, 'yRel': 3.319999933242798, 'status': 0.0}, {'aRel': 0.0, 'vRel': -1.4500000476837158, 'oncoming': False, 'stationary': False, 'trackID': 4146, 'dRel': 34.91999816894531, 'yRel': -3.4800000190734863, 'status': 0.0}, {'aRel': 0.0, 'vRel': -1.5, 'oncoming': False, 'stationary': False, 'trackID': 4085, 'dRel': 34.91999816894531, 'yRel': -3.5199999809265137, 'status': 0.0}, {'aRel': 0.0, 'vRel': -2.2249999046325684, 'oncoming': False, 'stationary': False, 'trackID': 4150, 'dRel': 63.0, 'yRel': -0.0, 'status': 0.0}, {'aRel': 0.0, 'vRel': -1.4500000476837158, 'oncoming': False, 'stationary': False, 'trackID': 4151, 'dRel': 81.63999938964844, 'yRel': -0.0, 'status': 0.0}, {'aRel': 0.0, 'vRel': -1.850000023841858, 'oncoming': False, 'stationary': False, 'trackID': 4156, 'dRel': 57.52000045776367, 'yRel': -3.200000047683716, 'status': 0.0}], [{'aRel': 0.0, 'vRel': -2.299999952316284, 'oncoming': False, 'stationary': False, 'trackID': 4164, 'dRel': 62.599998474121094, 'yRel': 0.03999999910593033, 'status': 0.0}, {'aRel': 0.0, 'vRel': -2.0999999046325684, 'oncoming': False, 'stationary': False, 'trackID': 4165, 'dRel': 63.560001373291016, 'yRel': -0.0, 'status': 0.0}, {'aRel': 0.0, 'vRel': 2.674999952316284, 'oncoming': False, 'stationary': False, 'trackID': 4166, 'dRel': 184.55999755859375, 'yRel': 3.119999885559082, 'status': 0.0}, {'aRel': 0.0, 'vRel': -1.850000023841858, 'oncoming': False, 'stationary': False, 'trackID': 4109, 'dRel': 57.439998626708984, 'yRel': -3.240000009536743, 'status': 0.0}, {'aRel': 0.0, 'vRel': -1.899999976158142, 'oncoming': False, 'stationary': False, 'trackID': 4134, 'dRel': 96.27999877929688, 'yRel': -3.4000000953674316, 'status': 0.0}, {'aRel': 0.0, 'vRel': 5.925000190734863, 'oncoming': False, 'stationary': False, 'trackID': 4013, 'dRel': 67.4800033569336, 'yRel': 3.319999933242798, 'status': 0.0}, {'aRel': 0.0, 'vRel': -1.4500000476837158, 'oncoming': False, 'stationary': False, 'trackID': 4146, 'dRel': 34.84000015258789, 'yRel': -3.5199999809265137, 'status': 0.0}, {'aRel': 0.0, 'vRel': -1.5, 'oncoming': False, 'stationary': False, 'trackID': 4085, 'dRel': 34.84000015258789, 'yRel': -3.5199999809265137, 'status': 0.0}, {'aRel': 0.0, 'vRel': -2.2249999046325684, 'oncoming': False, 'stationary': False, 'trackID': 4150, 'dRel': 62.7599983215332, 'yRel': -0.0, 'status': 0.0}, {'aRel': 0.0, 'vRel': -1.399999976158142, 'oncoming': False, 'stationary': False, 'trackID': 4151, 'dRel': 81.55999755859375, 'yRel': 0.03999999910593033, 'status': 0.0}, {'aRel': 0.0, 'vRel': -1.8250000476837158, 'oncoming': False, 'stationary': False, 'trackID': 4156, 'dRel': 57.400001525878906, 'yRel': -3.319999933242798, 'status': 0.0}]], [[2, 0.800000011920929], [-2.5, 1.21920929], [6.85, 0.800000011920929], [0.0, -0.200000011920929]])


def normX(data, data_scale=None):
    if data_scale==None:
        #v_ego = [i[0] for i in data]
        v_ego = [i[0] for i in data]
        a_ego = [i[1] for i in data]
        
        scales = {'v_ego_scale': [min(v_ego), max(v_ego)],
                  'a_ego_scale': [min(a_ego), max(a_ego)]}

        normalized = [[interp_fast(i[0], scales['v_ego_scale']), interp_fast(i[1], scales['a_ego_scale'])] for i in data]
        #normalized = [[i[0], i[1]] for i in data]
        return normalized, scales
    else:
        y = [data_scale[0], data_scale[1]]
        return np.interp(data, y, scale)

def normXOld(data, data_scale=None):
    if data_scale==None:
        v = [inner for outer in data for inner in [outer[0]]+[outer[2]]] # all vel data
        #a = [inner for outer in data for inner in [outer[1]]+[outer[4]]] # all accel data
        a = [i[4] for i in data] # all lead accel data
        x = [i[3] for i in data] # all distance data
        
        v_scale = [min(v), max(v)]
        a_scale = [min(a), max(a)]
        x_scale = [min(x), max(x)]
        
        normalized = [[np.interp(i[0], v_scale, scale), np.interp(i[1], a_scale, scale), np.interp(i[2], v_scale, scale), np.interp(i[3], x_scale, scale), np.interp(i[4], a_scale, scale)] for i in data]
        #normalized = [[np.interp(i[0], v_scale, scale), np.interp(i[2], v_scale, scale), np.interp(i[3], x_scale, scale), np.interp(i[4], a_scale, scale)] for i in data]
        return {'v_scale': v_scale, 'a_scale': a_scale, 'x_scale': x_scale, 'normalized': np.array(normalized)}
    else:
        y = [data_scale[0], data_scale[1]]
        return np.interp(data, y, scale)


#a = normX([[9.387969017029, 1.137865662575, 12.12175655365, 30.25, 3.817160964012, 1], [18.121639251709, -0.024567155167, 19.496248245239, 14.0, 0.072589494288, 0]])
#print(a)