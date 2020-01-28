from scipy.integrate import odeint
import matplotlib.pyplot as plt

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
    m = 500      # vehicle mass (kg)
    # calculate derivative of the velocity
    dv_dt = (1.0/(m+load)) * (Fp*u - 0.5*rho*Cd*A*v**2)
    return dv_dt

v = 10  # starting velocity
delta_t = .1  # time in between timesteps
u = 20  # gas pedal position, -50 to 100
load = 0
a = []
for i in range(100):
    v_prev = v
    v = odeint(vehicle,v_prev,[0,delta_t],args=(u,load))[-1] # returns next velocity
    cur_acc = (v - v_prev) / delta_t
    print(cur_acc)
    #print(v)
    a.append(v)

plt.plot(range(len(a)), [i[-1] for i in a])
plt.show()