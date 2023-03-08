import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
import numpy as np
from sklearn import linear_model

from functools import partial
from pysindy.feature_library import CustomLibrary
from sklearn.preprocessing import PolynomialFeatures
import pysindy as ps
import warnings
import random

from pysindy.differentiation import SmoothedFiniteDifference
from torchdiffeq import odeint


def update_coef(
    mask_threshold, degree, coef, mask, train_X, train_T, optimizer, abs_max
):

    error = predict_for_train(
        degree,
        train_X,
        train_T,
        ((torch.sigmoid(mask) > mask_threshold).float() * coef),
        abs_max,
    )
    cost = torch.sum(error)
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    return coef, cost


def update_mask(
    degree,
    mask_schedule,
    reg,
    env_num,
    coef,
    mask,
    train_X,
    train_T,
    mask_optimizer,
    epo,
    abs_max,
):
    mask_wo_reg = predict_for_valid(
        degree, train_X, train_T, coef * torch.sigmoid(mask), abs_max
    ).sum()
    mask_cost = mask_wo_reg + env_num * reg * (
        (1.0 + mask_schedule) ** ((epo - 1))
    ) * torch.norm(torch.sigmoid(mask), p=1)

    mask_optimizer.zero_grad()
    mask_cost.backward()
    mask_optimizer.step()

    return mask


def print_mask_info(mask_threshold, mask):
    sparsity = (((mask > mask_threshold).float() == 0.0).float().sum()) / (
        mask.shape[0] * mask.shape[1]
    )
    print("sparsity={}".format(sparsity))
    print(mask)


def print_RMSE(mask_threshold, degree, epoch, coef, mask, train_X, train_T, abs_max):
    message = "epoch " + str(epoch) + ")"
    error = predict_for_train(
        degree,
        train_X,
        train_T,
        ((torch.sigmoid(mask) > mask_threshold).float() * coef),
        abs_max,
    )

    for env in range(error.shape[0]):
        message += "\tenv[" + str(env) + "]: " + str(error[env].item())
    print(message)


def cal_err(mask_threshold, degree, coef, mask, X_data, eval_time, abs_max):
    return predict_for_valid(
        degree,
        X_data,
        eval_time,
        ((torch.sigmoid(mask) > mask_threshold).float() * coef),
        abs_max,
    ).mean()


def cal_mask_perf(data_type, mask_threshold, mask):
    mask_gt = get_mask_gt(data_type, mask)
    precision = (
        torch.logical_and(mask_gt, (mask > mask_threshold).float()).float().sum()
        / (mask > mask_threshold).float().sum()
    )
    recall = (
        torch.logical_and(mask_gt, (mask > mask_threshold).float()).float().sum()
        / mask_gt.sum()
    )
    return precision, recall


def get_mask_gt(data_type, mask):
    mask_gt = torch.zeros_like(mask)
    if data_type == "linear":
        mask_gt[1, 0] = 1.0
        mask_gt[1, 1] = 1.0
        mask_gt[2, 0] = 1.0
        mask_gt[2, 1] = 1.0
        mask_gt[3, 2] = 1.0
    elif data_type == "lorenz":
        mask_gt[1, 0] = 1.0
        mask_gt[1, 1] = 1.0
        mask_gt[2, 0] = 1.0
        mask_gt[2, 1] = 1.0
        mask_gt[3, 2] = 1.0
        mask_gt[6, 1] = 1.0
        mask_gt[5, 2] = 1.0
    elif data_type == "lotka":
        mask_gt[1, 0] = 1.0
        mask_gt[2, 1] = 1.0
        mask_gt[4, 0] = 1.0
        mask_gt[4, 1] = 1.0
    elif data_type == "pendulum":
        mask_gt[2, 0] = 1.0
        mask_gt[2, 1] = 1.0
        mask_gt[-3, 1] = 1.0
    else:
        raise NotImplementedError

    return mask_gt


def predict_for_train(degree, X_data, eval_time, masked_coef, abs_max):

    # Integrator keywords for solve_ivp
    integrator_keywords = {}
    integrator_keywords["method"] = "rk4"
    # integrator_keywords["adjoint_params"] = ()

    loss = torch.zeros(masked_coef.shape[0]).to("cuda")

    for env_idx in range(masked_coef.shape[0]):
        for i in range(len(eval_time) - 1):
            x_train = odeint(
                partial(
                    predicted_func_clip,
                    param=masked_coef[env_idx],
                    deg=degree,
                    abs_max=abs_max[env_idx],
                ),
                X_data[:, env_idx, i, :],
                eval_time[i : i + 2],
                **integrator_keywords
            )
            loss[env_idx] += F.smooth_l1_loss(
                x_train.permute(1, 0, 2),
                X_data[:, env_idx, i : i + 2, :],
                reduction="sum",
            )

    traj, env, time, state = X_data.shape

    return loss / (traj * time * state)


def predicted_func_clip(t, x, param, deg, abs_max):
    poly = PolynomialFeatures(deg)

    ### pre processing for preserving NaN ###
    if (torch.abs(x).detach().cpu().numpy() > 5000).any():
        return x / 100

    x = x.detach().cpu()
    phi = poly.fit_transform(x)

    if param.shape[0] == (phi.shape[1] + 3):
        functions = [lambda x: np.sin(x), lambda x, y: np.sin(x + y)]
        sin = CustomLibrary(library_functions=functions)
        add_phi = sin.fit_transform(x)
        phi = np.concatenate((phi, add_phi), 1)

    phi = phi / abs_max

    return torch.matmul(torch.Tensor(phi).to("cuda"), torch.Tensor(param).to("cuda"))


def predict_for_valid(degree, gt, eval_time, masked_coef, abs_max):

    # Integrator keywords for solve_ivp
    integrator_keywords = {}
    integrator_keywords["method"] = "rk4"
    ##############################
    loss = torch.zeros(masked_coef.shape[0]).to("cuda")
    for env_idx in range(masked_coef.shape[0]):
        x_train = odeint(
            partial(
                predicted_func_clip,
                param=masked_coef[env_idx],
                deg=degree,
                abs_max=abs_max[env_idx],
            ),
            gt[:, env_idx, 0, :],
            eval_time,
            **integrator_keywords
        )
        loss[env_idx] += F.mse_loss(
            x_train.permute(1, 0, 2), gt[:, env_idx, :, :], reduction="sum"
        )

    traj, env, time, state = gt.shape

    return loss / (traj * time * state)


def init_by_SINDY(
    data_type, degree, dt, X, train_T, val_X, val_T, abs_max, train_X_dots
):

    best = float("inf")
    coef = None

    coef_per_env = []
    X = X.permute(1, 0, 2, 3)
    X = X.detach().cpu().numpy()

    train_X_dots = train_X_dots.permute(1, 0, 2, 3).cpu().numpy()

    state = X.shape[-1]
    integrator_keywords = {}
    integrator_keywords["method"] = "rk4"

    for env, train_X in enumerate(X):
        train_X_dot = train_X_dots[env]

        train_X = train_X.reshape(-1, state)
        train_X_dot = train_X_dot.reshape(-1, state)

        for threshold in [0.1, 0.05, 0.01]:
            for alpha in [1, 0.5, 0.1, 0.05]:
                if data_type == "pendulum":
                    lb_p = ps.PolynomialLibrary(degree=degree)
                    custom = [lambda x: np.sin(x), lambda x, y: np.sin(x + y)]
                    sin = ps.CustomLibrary(library_functions=custom)
                    lb = ps.GeneralizedLibrary([lb_p, sin])
                else:
                    lb = ps.PolynomialLibrary(degree=degree)

                model = ps.SINDy(
                    optimizer=ps.STLSQ(threshold=threshold, alpha=alpha),
                    feature_library=lb,
                )
                model.fit(train_X, t=dt, x_dot=train_X_dot)

                pred = odeint(
                    partial(
                        predicted_func_clip,
                        param=torch.tensor(
                            model.coefficients().T
                            * np.expand_dims(abs_max[env], axis=-1)
                        )
                        .to("cuda")
                        .float(),
                        deg=degree,
                        abs_max=abs_max[env],
                    ),
                    val_X[:, env, 0, :],
                    val_T,
                    **integrator_keywords
                )

                val_loss = F.mse_loss(pred.permute(1, 0, 2), val_X[:, env, :, :])
                if best > val_loss:
                    coef = model.coefficients().T

        coef_per_env.append(coef)

    return torch.tensor(np.array(coef_per_env) * np.expand_dims(abs_max, axis=-1))


def init_mask(coef):
    sigmoid_mask = 0.7 * torch.sum((coef != 0).float(), dim=0) / coef.shape[0]
    mask = torch.special.logit(sigmoid_mask)

    return mask
