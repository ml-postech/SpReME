import numpy as np
from scipy.integrate import solve_ivp
from pysindy import SINDy
lorenz = lambda t,z : [10*(z[1] - z[0]),
                        z[0]*(28 - z[2]) - z[1],
                        z[0]*z[1] - 8/3*z[2]]
t = np.arange(0,2,.002)

integrator_keywords = {}
integrator_keywords["method"] = "LSODA"





x_train = solve_ivp(
        lorenz,
        (t[0], t[-1]),
        [8., -8., 27.],
        t_eval=t,
        **integrator_keywords
).y.T