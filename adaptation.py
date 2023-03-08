import torch
import torch.nn as nn

import numpy as np

from pysindy.differentiation import SmoothedFiniteDifference
import argparse
from generate_data import gen_data, gen_test, PHY_dataset, cal_abs_max
import torch.optim as optim
from functions import predicted_func, predicted_func_clip, cal_err2, update_coef2

from torchdiffeq import odeint
from functools import partial
from torch.nn import functional as F
import copy


def adaptation(
    mask,
    coef,
    adpt_X,
    adpt_T,
    valid_X,
    valid_T,
    abs_max,
    integrator_keywords,
    optimizer,
    data_type,
):

    min_err = torch.tensor(float("inf"))
    best_coef = None
    for epoch in range(10000):
        loss = torch.tensor(0.0).float().to("cuda")
        for i in range(adpt_T.shape[0] - 1):
            x_train = (
                odeint(
                    partial(
                        predicted_func_clip,
                        param=mask * coef.to("cuda").float(),
                        deg=args.degree,
                        abs_max=abs_max,
                    ),
                    adpt_X[i : i + 1],
                    adpt_T[i : i + 2],
                    **integrator_keywords
                )
                .permute(1, 0, 2)
                .squeeze()
            )
            loss += F.smooth_l1_loss(x_train, adpt_X[i : i + 2])

        loss /= adpt_T.shape[0]
        print(epoch, "] ", loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred = (
            odeint(
                partial(
                    predicted_func_clip,
                    param=mask * coef.to("cuda").float(),
                    deg=args.degree,
                    abs_max=abs_max,
                ),
                valid_X[0:1],
                valid_T,
                **integrator_keywords
            )
            .permute(1, 0, 2)
            .squeeze()
        )

        err = F.mse_loss(pred, valid_X)
        if err < min_err:
            min_err = err
            best_coef = copy.deepcopy(coef)
            torch.save(best_coef, data_type + "adpted_coef_.pt")

    return best_coef


def test_prediction(coef, test_X, test_T, integrator_keywords, degree):
    pred = (
        odeint(
            partial(
                predicted_func, param=coef.to("cuda").float(), deg=degree, abs_max=1
            ),
            test_X[:, 0, :],
            test_T[0],
            **integrator_keywords
        )
        .permute(1, 0, 2)
        .squeeze()
    )

    loss = F.mse_loss(pred, test_X, reduction="none")
    loss_mean = F.mse_loss(pred, test_X)

    loss = loss.mean(dim=(1, 2))
    loss_max = torch.max(loss)
    loss_min = torch.min(loss)

    print("mean: ", loss_mean)
    print("min: ", loss_min)
    print("max: ", loss_max)

    return loss


def main():
    data_type = args.data_type
    time = args.time
    dt = args.dt
    degree = args.degree
    traj_num = args.traj_num

    adaptation_param = {
        "data_type": data_type,
        "time": time,
        "dt": dt,
        "traj_num": 1,
        "env_num": 1,
        "env_var": 0.0,
        "seed": 100,
        "degree": degree,
        "adaptation": True,
    }

    params, x0_trains, m, n, adpt_X, adpt_T = gen_data(**adaptation_param)
    test_X, test_T = gen_test(params, traj_num, time*2.5, dt/2, data_type)

    val_len = adpt_X.shape[2] // 5
    val_X = torch.tensor(adpt_X[:, :, -val_len - 1 :])
    val_T = torch.tensor(adpt_T[:, -val_len - 1 :])

    adpt_X = adpt_X[:, :, :-val_len]
    adpt_T = adpt_T[:, :-val_len]

    abs_max, _ = cal_abs_max(args, adpt_X)

    adpt_X = adpt_X.squeeze().to("cuda")
    adpt_T = adpt_T.squeeze().to("cuda")

    val_X = val_X.squeeze().to("cuda")
    val_T = val_T.squeeze().to("cuda")

    test_X = test_X.squeeze().to("cuda")
    test_T = test_T.squeeze().to("cuda")

    print(adpt_X.shape, test_X.shape)

    mask = torch.load(data_type + "_best_mask.pt")  ######################
    integrator_keywords = {}
    integrator_keywords["method"] = "rk4"
    coef = nn.Parameter(torch.randn(mask.shape).float().to("cuda") * mask)
    optimizer = optim.Adam([coef], lr=args.lr)

    adpt_coef = adaptaion(
        mask,
        coef,
        adpt_X,
        adpt_T,
        valid_X,
        valid_T,
        abs_max,
        integrator_keywords,
        optimizer,
        data_type,
    )
    loss = test_prediction(adpt_coef, test_X, test_T, integrator_keywords, degree)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--data-type", type=str, default="lorenz", help="data-type (3d or lotka or lorenz or pendulum)"
    )
    argparser.add_argument(
        "--degree", type=int, default=5, metavar="S", help="poly degree"
    )
    argparser.add_argument(
        "--dt", type=float, help="interval of train data", default=0.02
    )
    argparser.add_argument("--time", type=int, help="time length", default=4)
    argparser.add_argument("--traj-num", type=int, default=16, metavar="S", help="")
    argparser.add_argument(
        "--lr", type=float, default=0.01, metavar="S", help="learning rate"
    )
    args = argparser.parse_args()

    main()
