import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.cm import rainbow
import numpy as np
from scipy.integrate import solve_ivp
from scipy.io import loadmat
from pysindy.utils import linear_damped_SHO
from pysindy.utils import cubic_damped_SHO
from pysindy.utils import linear_3D
from pysindy.utils import hopf
from pysindy.utils import lorenz

import pysindy as ps

# ignore user warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

np.random.seed(1000)  # Seed for reproducibility

# Integrator keywords for solve_ivp
integrator_keywords = {}
integrator_keywords['rtol'] = 1e-12
integrator_keywords['method'] = 'LSODA'
integrator_keywords['atol'] = 1e-12

# Generate training data

dt = .01
t_train = np.arange(0, 50, dt)
t_train_span = (t_train[0], t_train[-1])
x0_train = [2, 0, 1]
x_train = solve_ivp(linear_3D, t_train_span, 
                    x0_train, t_eval=t_train, **integrator_keywords).y.T

# Fit the model

poly_order = 5
threshold = 0.01

model = ps.SINDy(
    optimizer=ps.STLSQ(threshold=threshold),
    feature_library=ps.PolynomialLibrary(degree=poly_order)
)
model.fit(x_train, t=dt)
model.print()

# # Generate training data

# dt = 0.001
# t_train = np.arange(0, 100, dt)
# t_train_span = (t_train[0], t_train[-1])
# x0_train = [-8, 8, 27]
# x_train = solve_ivp(lorenz, t_train_span, 
#                     x0_train, t_eval=t_train, **integrator_keywords).y.T
# x_dot_train_measured = np.array(
#     [lorenz(0, x_train[i]) for i in range(t_train.size)]
# )

# # Fit the models and simulate

# poly_order = 5
# threshold = 0.05

# noise_levels = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]

# models = []
# t_sim = np.arange(0, 20, dt)
# x_sim = []
# for eps in noise_levels:
#     model = ps.SINDy(
#         optimizer=ps.STLSQ(threshold=threshold),
#         feature_library=ps.PolynomialLibrary(degree=poly_order),
#     )
#     # import pdb; pdb.set_trace()
#     model.fit(
#         x_train,
#         t=dt,
#         x_dot=x_dot_train_measured
#         + np.random.normal(scale=eps, size=x_train.shape),
#         quiet=True,
#     )
#     models.append(model)
#     x_sim.append(model.simulate(x_train[0], t_sim))