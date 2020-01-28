from scipy.integrate import odeint
import matplotlib.pyplot as plt
import numpy as np

def vehicle(v,t,u,load):
    # inputs
    #  v    = vehicle velocity (m/s)
    #  t    = time (sec)
    #  u    = gas pedal position (-50% to 100%)
    #  load = passenger load + cargo (kg)
    Cd = 0.24    # drag coefficient
    rho = 1.225  # air density (kg/m^3)
    A = 5.0      # cross-sectional area (m^2)
    Fp = 30      # thrust parameter (N/%pedal)
    m = 1301.81      # vehicle mass (kg)
    # calculate derivative of the velocity
    dv_dt = (1.0/(m+load)) * (Fp*u - 0.5*rho*Cd*A*v**2)
    return dv_dt


# for i in range(100):
#     v_prev = v
#     v = odeint(vehicle,v_prev,[0,delta_t],args=(u,load))[-1] # returns next velocity
#     cur_acc = (v - v_prev) / delta_t
#     print(cur_acc)
#     #print(v)
#     a.append(v)

#plt.plot(range(len(a)), [i[-1] for i in a])
#plt.show()

#gas = 0.0


v = [0]  # starting velocity
delta_t = .1  # time in between timesteps
gas = .75  # gas pedal position, -50 to 100
load = 130
#a = []

P = .3

steps = 100
errors = []
accels = []
gs = []
vs = []

'''des_acc = np.zeros(steps)
des_acc[len(des_acc) // 20:] = .5
des_acc[len(des_acc) // 5:] = 1.5
des_acc[int(len(des_acc) // 4):] = 0
des_acc[int(len(des_acc) // 3):] = .5
des_acc[int(len(des_acc) // 2):] = -1.25
des_acc[int(len(des_acc) // 1.5):] = .234'''
# des_acc = np.concatenate((np.linspace(0, 0, int(steps*.1)),
#                           np.linspace(0, .4, int(steps*.2)),
#                           np.linspace(.5, 1.2, int(steps*.1)),
#                           np.linspace(1.3, 1.4, int(steps*.15)),
#                           np.linspace(.9, -2.1, int(steps*.25)),
#                           np.linspace(.6, -1.1, int(steps*.2))))

des_acc = np.concatenate((np.linspace(0, 0, int(steps*.25)),
                          np.linspace(2, 2, int(steps*.25)),
                          np.linspace(-1.5, -1.5, int(steps*.25)),
                          np.linspace(.3, .3, int(steps*.25))))

cur_acc = 0.0
for idx, i in enumerate(range(steps)):
    #print(error)
    v_prev = v
    print(cur_acc)
    error = cur_acc - des_acc[idx]
    if error != 0:
        is_neg = error < 0
        error = ((abs(error) + 1) ** 1.2) - 1
        error = -error if is_neg else error
    print(error)
    gas = np.clip(-(error * .3), -1, 1)
    print(gas)
    v = odeint(vehicle, v_prev, [0, delta_t], args=(np.interp(gas, [-1, 0, 1], [-100, 0, 100]), load))[-1] # returns next velocity
    #print(v[0])
    cur_acc = (v[-1] - v_prev[-1]) / delta_t
    print()
    errors.append(error)
    vs.append(v[-1])
    
    #print(gas)
    gs.append(gas)
    #print()
    accels.append(cur_acc)
    if v[-1] < 0.0:
        break
    
    #print(gas)

steps = len(vs)
plt.clf()
plt.plot(range(steps), des_acc, linestyle='--', label='desired acc')
plt.plot(range(steps), accels, linestyle='--', label='actual acc')
#plt.plot(range(steps), gs, label='gas')
#plt.plot(range(steps), vs, label='v_ego')
#plt.plot(range(steps), errors, label='error')
plt.legend()
plt.title('This is {} seconds'.format(round(steps * delta_t , 4)))
plt.show()