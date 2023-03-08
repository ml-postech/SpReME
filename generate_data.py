import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import Dataset

import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import comb
from pysindy.utils import linear_damped_SHO, lorenz, linear_3D, lotka, pendulum, g_osci
from pysindy.utils import concat_sample_axis
from pysindy.utils import AxesArray
from pysindy.utils import comprehend_axes
from pysindy.utils import validate_no_reshape
from functools import partial
from pysindy.differentiation import FiniteDifference
from pysindy.feature_library import PolynomialLibrary
from pysindy.feature_library import CustomLibrary
from itertools import product
from typing import Collection
from typing import Sequence
from sklearn.preprocessing import PolynomialFeatures

from torchdiffeq import odeint


import pysindy as ps

# ignore user warnings
import warnings
import random
import argparse


class PHY_dataset(Dataset):
    def __init__(self, X, Y):
        super(PHY_dataset, self).__init__()

        self.X_data = X
        self.Y_data = Y

    def __getitem__(self, index):
        return self.X_data[index], self.Y_data[index]

    def __len__(self):
        return len(self.X_data)


def _zip_like_sequence(x, t):
    """Create an iterable like zip(x, t), but works if t is scalar."""
    if isinstance(t, Sequence):
        return zip(x, t)
    else:
        return product(x, [t])


def comprehend_and_validate(arr, t, feature_library):
    arr = AxesArray(arr, comprehend_axes(arr))
    arr = feature_library.correct_shape(arr)
    return validate_no_reshape(arr, t)


def predicted_func(t, x, param, deg, abs_max):
    poly = PolynomialFeatures(deg)
    x = x.detach().cpu()
    phi = poly.fit_transform(x)

    if param.shape[0] == (phi.shape[1] + 3):
        functions = [lambda x: np.sin(x), lambda x, y: np.sin(x + y)]
        sin = CustomLibrary(library_functions=functions)
        add_phi = sin.fit_transform(x)
        phi = np.concatenate((phi, add_phi), 1)

    phi = phi / abs_max

    return torch.matmul(torch.Tensor(phi).to("cuda"), torch.Tensor(param).to("cuda"))


def gen_data(
    data_type,
    time,
    dt,
    traj_num,
    env_num,
    env_var,
    degree,
    seed,
    adaptation=False,
):
    warnings.filterwarnings("ignore", category=UserWarning)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if adaptation:
        env_var = 0.0
        traj_num = 1
        env_num = 1

    print(env_num, time, dt)
    # Integrator keywords for solve_ivp
    integrator_keywords = {}
    integrator_keywords["rtol"] = 1e-12
    integrator_keywords["method"] = "LSODA"
    integrator_keywords["atol"] = 1e-12

    poly_order = degree
    poly = PolynomialFeatures(poly_order)
    if data_type == "pendulum":
        functions = [lambda x: np.sin(x), lambda x, y: np.sin(x + y)]
        sin = CustomLibrary(library_functions=functions)
    feature_library = ps.PolynomialLibrary(degree=poly_order)
    differentiation_method = FiniteDifference(axis=-2)

    # Generate training data
    t_train = np.arange(0, time, dt)
    t_train_span = (t_train[0], t_train[-1])

    if data_type == "linear":
        x0_trains = np.random.randn(traj_num, 3)
        if not adaptation:
            params = np.array([-0.1, 2.0, -2.0, -0.1, -0.3]) + (
                env_var**0.5
            ) * np.random.randn(env_num, 5)
        else:
            params = np.array([[-0.1, 2.0, -2.0, -0.1, -0.3]])
        function = linear_3D

    elif data_type == "lorenz":
        x0_trains = np.random.randn(traj_num, 3)
        if not adaptation:
            params = np.array([10, 28, 8 / 3]) + (env_var**0.5) * np.random.randn(
                env_num, 3
            )
        else:
            params = np.array([[10, 28, 8 / 3]])
        function = lorenz

    elif data_type == "lotka":
        if not adaptation:
            params = np.array(
                [
                    [0.5, 0.5, 0.5, 0.5],
                    [0.5, 0.75, 0.5, 0.5],
                    [0.5, 1.0, 0.5, 0.5],
                    [0.5, 0.5, 0.5, 0.75],
                    [0.5, 0.5, 0.5, 1.0],
                    [0.5, 0.75, 0.5, 0.75],
                    [0.5, 0.75, 0.5, 1.0],
                    [0.5, 1.0, 0.5, 0.75],
                    [0.5, 1.0, 0.5, 1.0],
                ]
            )
        else:
            params = np.array([[0.5, 0.625, 0.5, 1.125]])

        x0_trains = np.random.random((traj_num, 2)) + 1.0
        function = lotka
    elif data_type == "pendulum":
        if not adaptation:
            params = np.array([0.6 / 1.2, 9.8 / 10.0]) + (
                env_var**0.5
            ) * np.random.randn(env_num, 2)
        else:
            params = np.array([[0.6 / 1.2, 9.8 / 10.0]])

        x0_trains = np.random.randn(traj_num, 2) * 2.0 - 1
        radius = np.random.rand(traj_num, 1) + 1.3
        x0_trains = (
            x0_trains / np.sqrt((x0_trains**2).sum(axis=1, keepdims=True)) * radius
        )
        function = pendulum

    else:
        raise NotImplementedError("{} is not implemented data type".format(data_type))

    x0_trains = x0_trains.astype(np.float16).tolist()
    params = params.astype(np.float16).tolist()

    print("params=\n{}".format(params))
    x_stack = []
    t_stack = []

    for x0_train in x0_trains:
        x_stack_env = []
        for param in params:
            x_train = solve_ivp(
                partial(function, p=param),
                t_train_span,
                x0_train,
                t_eval=t_train,
                **integrator_keywords
            ).y.T
            x_stack_env.append(x_train)
            x_train = [
                comprehend_and_validate(xi, ti, feature_library)
                for xi, ti in _zip_like_sequence(x_train, dt)
            ]

        x_stack.append(np.array(x_stack_env))
        t_stack.append(np.array(t_train))

    train_X = torch.tensor(x_stack).float()

    train_T = torch.tensor(t_stack).float()

    traj_num = train_X.shape[0]
    m = train_X.shape[2]
    n = train_X.shape[-1]

    print(train_X.shape, train_T.shape)

    print(
        "data generated for {} environments. \n{} trajectory generated for each env.\ntime stamp number: {}\nstate number: {}".format(
            env_num, traj_num, m, n
        )
    )
    # gen_test

    return params, x0_trains, m, n, train_X, train_T


def cal_abs_max(data_type, degree, train_X):
    train_X = train_X.permute(1, 0, 2, 3)
    shape = train_X.shape
    train_X = train_X.reshape(-1, shape[-1])

    poly = PolynomialFeatures(degree)
    phi = poly.fit_transform(train_X)  ##shape: [time_stamp, candidate_num]
    if data_type == "pendulum":
        functions = [lambda x: np.sin(x), lambda x, y: np.sin(x + y)]
        sin = CustomLibrary(library_functions=functions)
        add_phi = sin.fit_transform(train_X)
        phi = np.concatenate((phi, add_phi), 1)

    phi = phi.reshape(shape[0], -1, phi.shape[-1])
    abs_max = np.max(np.abs(phi.reshape(shape[0], -1, phi.shape[-1])), 1)

    return abs_max, phi


def gen_test(params, traj_num, time, dt, data_type):
    integrator_keywords = {}
    integrator_keywords["rtol"] = 1e-12
    integrator_keywords["method"] = "LSODA"
    integrator_keywords["atol"] = 1e-12
    t_train = np.arange(0, time, dt)
    t_train_span = (t_train[0], t_train[-1])
    if data_type == "linear":
        x0_test = np.random.randn(traj_num, 3)
        function = linear_3D
    elif data_type == "lorenz":
        x0_test = np.random.randn(traj_num, 3)
        function = lorenz
    elif data_type == "lotka":
        x0_test = np.random.random((traj_num, 2)) + 1.0
        function = lotka
    elif data_type == "pendulum":
        x0_test = np.random.randn(traj_num, 2) * 2.0 - 1
        radius = np.random.rand(traj_num, 1) + 1.3
        x0_test = x0_test / np.sqrt((x0_test**2).sum(axis=1, keepdims=True)) * radius
        function = pendulum

    x_stack = []
    t_stack = []
    for x0_train in x0_test:

        x_stack_env = []
        for param in params:
            x_train = solve_ivp(
                partial(function, p=param),
                t_train_span,
                x0_train,
                t_eval=t_train,
                **integrator_keywords
            ).y.T

            x_stack_env.append(x_train.T)

        x_stack.append(np.array(x_stack_env))
        t_stack.append(t_train)

    train_X = torch.tensor(x_stack).float()
    train_T = torch.tensor(t_stack).float()

    return train_X, train_T
