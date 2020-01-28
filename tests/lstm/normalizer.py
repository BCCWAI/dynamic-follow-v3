import numpy as np
scale = [0.0, 1.0]

'''def interp(x, xp, fp):
  N = len(xp)
  def get_interp(xv):
    hi = 0
    while hi < N and xv > xp[hi]:
      hi += 1
    low = hi - 1
    return fp[-1] if hi == N and xv > xp[low] else (
      fp[0] if hi == 0 else 
      (xv - xp[low]) * (fp[hi] - fp[low]) / (xp[hi] - xp[low]) + fp[low])
  return [get_interp(v) for v in x] if hasattr(
    x, '__iter__') else get_interp(x)'''

'''def interp(x, xp, fp):
    return (((x - xp[0]) * (fp[1] - fp[0])) / (xp[1] - xp[0])) + fp[0]'''

def interp(x, min_max):
    return (x - min_max[0]) / (min_max[1] - min_max[0])

def norm(data, data_scale=None):
    time_to_comp = 22.9397234916687
    samps = 30000000
    if data_scale==None:
        v_ego = np.take(data, indices=0, axis=2)
        v_lead = np.take(data, indices=1, axis=2)
        x_lead = np.take(data, indices=2, axis=2)
        #v_ego = [inner[0] for outer in data for inner in outer]
        #v_lead = [inner[1] for outer in data for inner in outer]
        #x_lead = [inner[2] for outer in data for inner in outer]
        #for i in data:
            #for x in i:
                #all_v.append(x[0])
                #all_v.append(x[2])
                #all_a.append(x[1])
                #all_a.append(x[4])
                #all_x.append(x[3])
        scales = {     
                'v_ego_scale': [np.amin(v_ego), np.amax(v_ego)],
                'v_lead_scale': [np.amin(v_lead), np.amax(v_lead)],
                'x_lead_scale': [np.amin(x_lead), np.amax(x_lead)]
                }
        
        nums = np.prod(data.shape)
        time_to_comp = (time_to_comp * nums) / samps
        print('Normalization should take about {} minutes!'.format(round(time_to_comp / 60, 3)))
        
        #normalized = [[[interp(d[0], v_scale, scale), interp(d[1], a_scale, scale), interp(d[2], v_scale, scale), interp(d[3], x_scale, scale), interp(d[4], a_scale, scale)] for d in i] for i in data]
        #normalized = [[[interp(d[0], scales['v_ego_scale'], scale), interp(d[1], scales['v_lead_scale'], scale), interp(d[2], scales['x_lead_scale'], scale)] for d in i] for i in data]
        normalized = [[[interp(d[0], scales['v_ego_scale']), interp(d[1], scales['v_lead_scale']), interp(d[2], scales['x_lead_scale'])] for d in i] for i in data]
        return {'scales': scales, 'normalized': np.array(normalized)}
    else:
        y = [data_scale[0], data_scale[1]]
        return interp(data, y, scale)