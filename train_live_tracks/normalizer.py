import numpy as np
import functools
import operator

scale = [0.0, 1.0]  # wider scale might improve accuracy


def interp_fast(x, min_max):
    return (x - min_max[0]) / (min_max[1] - min_max[0])


def normX(track_samples, car_data):
    yRel = functools.reduce(operator.iconcat, [[track['yRel'] for track in line] for line in track_samples], [])
    dRel = functools.reduce(operator.iconcat, [[track['dRel'] for track in line] for line in track_samples], [])
    vRel = functools.reduce(operator.iconcat, [[track['vRel'] for track in line] for line in track_samples], [])
    scales = {'yRel': [min(yRel), max(yRel)],
              'dRel': [min(dRel), max(dRel)],
              'vRel': [min(vRel), max(vRel)]}

    # normalize tracks
    tracks_normalized = []
    for idx, sample in enumerate(track_samples):
        tracks_normalized.append([])
        for track in sample:
            tracks_normalized[idx].append([interp_fast(track['yRel'], scales['yRel']),
                                           interp_fast(track['dRel'], scales['dRel']),
                                           interp_fast(track['vRel'], scales['vRel'])])
    # normalize car data
    car_data = np.array(car_data)
    mins = np.min(car_data, axis=0)
    maxs = np.max(car_data, axis=0)
    scales['v_ego'] = [mins[0], maxs[0]]
    scales['steer_angle'] = [mins[1], maxs[1]]
    scales['steer_rate'] = [mins[2], maxs[2]]
    scales['a_lead'] = [mins[3], maxs[3]]
    scales['x_lead'] = [mins[7], maxs[7]]
    scales['v_lead'] = [mins[8], maxs[8]]
    car_data_normalized = []
    for i in car_data:
        if i[6] == 0:
            i[3] = 0.0
            i[7] = 0.0
            i[8] = 0.0
        else:
            i[3] = interp_fast(i[3], scales['a_lead'])
            i[7] = interp_fast(i[7], scales['x_lead'])
            i[8] = interp_fast(i[8], scales['v_lead'])
        car_data_normalized.append([interp_fast(i[0], scales['v_ego']),
                                    interp_fast(i[1], scales['steer_angle']),
                                    interp_fast(i[2], scales['steer_rate']),
                                    i[3],
                                    # i[7],
                                    # i[8]])
                                    int(i[4]), int(i[5])])  # adds blinkers

    return tracks_normalized, car_data_normalized, scales

# tracks_normalized, car_data_normalized, scales = normTracks([[{'aRel': 0.0, 'vRel': -2.3499999046325684, 'oncoming': False, 'stationary': False, 'trackID': 4164, 'dRel': 62.79999923706055, 'yRel': 0.07999999821186066, 'status': 0.0}, {'aRel': 0.0, 'vRel': -2.1500000953674316, 'oncoming': False, 'stationary': False, 'trackID': 4165, 'dRel': 64.12000274658203, 'yRel': -0.03999999910593033, 'status': 0.0}, {'aRel': 0.0, 'vRel': 2.549999952316284, 'oncoming': False, 'stationary': False, 'trackID': 4166, 'dRel': 184.0399932861328, 'yRel': 3.0799999237060547, 'status': 0.0}, {'aRel': 0.0, 'vRel': -1.875, 'oncoming': False, 'stationary': False, 'trackID': 4109, 'dRel': 58.2400016784668, 'yRel': -3.0399999618530273, 'status': 0.0}, {'aRel': 0.0, 'vRel': -1.899999976158142, 'oncoming': False, 'stationary': False, 'trackID': 4134, 'dRel': 96.72000122070312, 'yRel': -3.359999895095825, 'status': 0.0}, {'aRel': 0.0, 'vRel': 5.875, 'oncoming': False, 'stationary': False, 'trackID': 4013, 'dRel': 66.31999969482422, 'yRel': 3.319999933242798, 'status': 0.0}, {'aRel': 0.0, 'vRel': -1.4500000476837158, 'oncoming': False, 'stationary': False, 'trackID': 4146, 'dRel': 35.20000076293945, 'yRel': -3.5199999809265137, 'status': 0.0}, {'aRel': 0.0, 'vRel': -1.5, 'oncoming': False, 'stationary': False, 'trackID': 4085, 'dRel': 35.20000076293945, 'yRel': -3.5999999046325684, 'status': 0.0}, {'aRel': 0.0, 'vRel': -2.200000047683716, 'oncoming': False, 'stationary': False, 'trackID': 4150, 'dRel': 64.04000091552734, 'yRel': -0.0, 'status': 0.0}, {'aRel': 0.0, 'vRel': -1.5, 'oncoming': False, 'stationary': False, 'trackID': 4151, 'dRel': 81.87999725341797, 'yRel': -0.0, 'status': 0.0}, {'aRel': 0.0, 'vRel': -1.975000023841858, 'oncoming': False, 'stationary': False, 'trackID': 4156, 'dRel': 57.79999923706055, 'yRel': -2.680000066757202, 'status': 0.0}], [{'aRel': 0.0, 'vRel': -2.4000000953674316, 'oncoming': False, 'stationary': False, 'trackID': 4164, 'dRel': 62.720001220703125, 'yRel': 0.07999999821186066, 'status': 0.0}, {'aRel': 0.0, 'vRel': -2.0999999046325684, 'oncoming': False, 'stationary': False, 'trackID': 4165, 'dRel': 63.68000030517578, 'yRel': -0.0, 'status': 0.0}, {'aRel': 0.0, 'vRel': 2.6500000953674316, 'oncoming': False, 'stationary': False, 'trackID': 4166, 'dRel': 184.47999572753906, 'yRel': 3.0799999237060547, 'status': 0.0}, {'aRel': 0.0, 'vRel': -1.850000023841858, 'oncoming': False, 'stationary': False, 'trackID': 4109, 'dRel': 57.560001373291016, 'yRel': -3.1600000858306885, 'status': 0.0}, {'aRel': 0.0, 'vRel': -1.899999976158142, 'oncoming': False, 'stationary': False, 'trackID': 4134, 'dRel': 96.4000015258789, 'yRel': -3.359999895095825, 'status': 0.0}, {'aRel': 0.0, 'vRel': 5.900000095367432, 'oncoming': False, 'stationary': False, 'trackID': 4013, 'dRel': 67.23999786376953, 'yRel': 3.319999933242798, 'status': 0.0}, {'aRel': 0.0, 'vRel': -1.4500000476837158, 'oncoming': False, 'stationary': False, 'trackID': 4146, 'dRel': 34.91999816894531, 'yRel': -3.4800000190734863, 'status': 0.0}, {'aRel': 0.0, 'vRel': -1.5, 'oncoming': False, 'stationary': False, 'trackID': 4085, 'dRel': 34.91999816894531, 'yRel': -3.5199999809265137, 'status': 0.0}, {'aRel': 0.0, 'vRel': -2.2249999046325684, 'oncoming': False, 'stationary': False, 'trackID': 4150, 'dRel': 63.0, 'yRel': -0.0, 'status': 0.0}, {'aRel': 0.0, 'vRel': -1.4500000476837158, 'oncoming': False, 'stationary': False, 'trackID': 4151, 'dRel': 81.63999938964844, 'yRel': -0.0, 'status': 0.0}, {'aRel': 0.0, 'vRel': -1.850000023841858, 'oncoming': False, 'stationary': False, 'trackID': 4156, 'dRel': 57.52000045776367, 'yRel': -3.200000047683716, 'status': 0.0}], [{'aRel': 0.0, 'vRel': -2.299999952316284, 'oncoming': False, 'stationary': False, 'trackID': 4164, 'dRel': 62.599998474121094, 'yRel': 0.03999999910593033, 'status': 0.0}, {'aRel': 0.0, 'vRel': -2.0999999046325684, 'oncoming': False, 'stationary': False, 'trackID': 4165, 'dRel': 63.560001373291016, 'yRel': -0.0, 'status': 0.0}, {'aRel': 0.0, 'vRel': 2.674999952316284, 'oncoming': False, 'stationary': False, 'trackID': 4166, 'dRel': 184.55999755859375, 'yRel': 3.119999885559082, 'status': 0.0}, {'aRel': 0.0, 'vRel': -1.850000023841858, 'oncoming': False, 'stationary': False, 'trackID': 4109, 'dRel': 57.439998626708984, 'yRel': -3.240000009536743, 'status': 0.0}, {'aRel': 0.0, 'vRel': -1.899999976158142, 'oncoming': False, 'stationary': False, 'trackID': 4134, 'dRel': 96.27999877929688, 'yRel': -3.4000000953674316, 'status': 0.0}, {'aRel': 0.0, 'vRel': 5.925000190734863, 'oncoming': False, 'stationary': False, 'trackID': 4013, 'dRel': 67.4800033569336, 'yRel': 3.319999933242798, 'status': 0.0}, {'aRel': 0.0, 'vRel': -1.4500000476837158, 'oncoming': False, 'stationary': False, 'trackID': 4146, 'dRel': 34.84000015258789, 'yRel': -3.5199999809265137, 'status': 0.0}, {'aRel': 0.0, 'vRel': -1.5, 'oncoming': False, 'stationary': False, 'trackID': 4085, 'dRel': 34.84000015258789, 'yRel': -3.5199999809265137, 'status': 0.0}, {'aRel': 0.0, 'vRel': -2.2249999046325684, 'oncoming': False, 'stationary': False, 'trackID': 4150, 'dRel': 62.7599983215332, 'yRel': -0.0, 'status': 0.0}, {'aRel': 0.0, 'vRel': -1.399999976158142, 'oncoming': False, 'stationary': False, 'trackID': 4151, 'dRel': 81.55999755859375, 'yRel': 0.03999999910593033, 'status': 0.0}, {'aRel': 0.0, 'vRel': -1.8250000476837158, 'oncoming': False, 'stationary': False, 'trackID': 4156, 'dRel': 57.400001525878906, 'yRel': -3.319999933242798, 'status': 0.0}]], [[2, 0.800000011920929], [-2.5, 1.21920929], [6.85, 0.800000011920929], [0.0, -0.200000011920929]])
