import wandb

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F

from generate_data import gen_data, predict_data
from tmp_mask_gen import mask_gen
from functions import (
    print_mask_info,
    update_coef,
    update_mask,
    print_RMSE,
    cal_err,
    cal_mask_perf,
)

import numpy as np
import matplotlib.pyplot as plt

import random
import warnings
from sklearn.model_selection import train_test_split


def test(args, device):
    warnings.filterwarnings("ignore", category=UserWarning)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if args.mask_type != 0:
        args.wandb = bool(args.wandb)
        args.lasso = bool(args.lasso)
        args.ste = bool(args.ste)

        if args.wandb:
            wandb.init(project="coef-adaptation")
            wandb.config.update(args)
            min_err = None

    args.reg = 0.0
    args.coef_lr = args.adpt_lr / 15

    params, x0_trains, m, p, n, test_X, test_Phi, test_X_dot, abs_max = gen_data(
        args, train_time=False
    )

    test_X = test_X.to(device)
    test_Phi = test_Phi.to(device)
    test_X_dot = test_X_dot.to(device)
    abs_max = abs_max.to(device)

    idx = list(range(0, int(10 / args.dt), 2))
    adapt_X = test_X[:, :, idx]
    adapt_Phi = test_Phi[:, :, idx]
    adapt_X_dot = test_X_dot[:, :, idx]

    if args.mask_type == 0:
        file_name = "best_mask of " + args.data_type + ".pt"
        meta_mask = torch.load(file_name).to("cpu")
        mask_type = 0
    else:
        mask_type = args.mask_type
        meta_mask = mask_gen(args, mask_type)

    coef = nn.init.xavier_uniform_(torch.empty(1, p, n))

    coef = coef * (meta_mask != 0.0).float()
    meta_mask = torch.where(meta_mask == 0.0, -1e10, 1e10)

    coef = coef.to(device)
    meta_mask = meta_mask.to(device)

    for epoch in range(1, args.adpt_epoch + 1):
        args.lasso = False
        coef, cost_w_reg = update_coef(args, coef, meta_mask, adapt_Phi, adapt_X_dot)

        message = "epoch " + str(epoch) + ")"
        pred_y = torch.matmul(adapt_Phi, coef)
        error = ((adapt_X_dot - pred_y) ** 2).sum(0)
        for env in range(error.shape[0]):
            message += (
                "\tenv[" + str(env) + "]: " + str(torch.mean(error).item() ** 0.5)
            )
        print(message)

        if args.wandb and args.mask_type != 0 and epoch % 10 == 1:
            err_per_env = (error.mean(0).mean() ** 0.5).item()
            if min_err is None:
                min_err = err_per_env
            if min_err > err_per_env:
                min_err = err_per_env
            wandb.log(
                {
                    "err": err_per_env,
                    "min_err": min_err,
                }
            )

    adapted_coef = coef / abs_max.unsqueeze(-1)

    predicted_X = predict_data(args, x0_trains, adapted_coef.cpu())
    predicted_X = predicted_X.to(device)
    print(predicted_X.shape)
    criterion = nn.MSELoss(reduce=False)
    loss = criterion(predicted_X, test_X)

    print(loss.shape)

    loss_plot = loss.mean(-1).mean(0).mean(0).cpu().detach().numpy()
    x_plot = np.array(list(range(0, loss_plot.shape[0]))) * args.dt

    fig = plt.figure(figsize=(8, 6))
    plt.plot(x_plot, loss_plot)
    loss_name = (
        "/data_seoul/mjeongp/SINDy_MASK/graph/loss_"
        + args.data_type
        + "_"
        + str(mask_type)
        + "_err_"
        + str(loss.sum().item())
        + ".png"
    )
    fig.savefig(loss_name)

    torch.save(test_X, "test_X.pt")
    torch.save(predicted_X, "predicted_X.pt")
    torch.save(loss, "loss.pt")

    min_env = torch.argmin(loss.sum(-1).sum(-1).sum(0))
    min_traj = torch.argmin(loss[:, min_env].sum(-1).sum(-1))

    s_gt = torch.t(test_X[min_traj, min_env]).cpu().detach().numpy()
    s_pd = torch.t(predicted_X[min_traj, min_env]).cpu().detach().numpy()

    gt_xline = s_gt[0]
    gt_yline = s_gt[1]
    if args.data_type == "lorenz" or args.data_type == "3d":
        gt_zline = s_gt[2]

    pd_x_in, pd_x_ex = train_test_split(s_pd[0], shuffle=False, test_size=0.8)
    pd_y_in, pd_y_ex = train_test_split(s_pd[1], shuffle=False, test_size=0.8)

    if args.data_type == "lorenz" or args.data_type == "3d":
        pd_z_in, pd_z_ex = train_test_split(s_pd[2], shuffle=False, test_size=0.8)

    fig = plt.figure(figsize=(20, 6))

    if args.data_type == "lorenz" or args.data_type == "3d":

        ax0 = fig.add_subplot(131, projection="3d")
        ax1 = fig.add_subplot(132, projection="3d")
        ax2 = fig.add_subplot(133, projection="3d")

        ax0.plot3D(gt_xline, gt_yline, gt_zline)

        ax1.plot3D(pd_x_in, pd_y_in, pd_z_in)
        ax1.plot3D(pd_x_ex, pd_y_ex, pd_z_ex, "red")

        ax2.plot3D(gt_xline, gt_yline, gt_zline)

        ax2.plot3D(pd_x_in, pd_y_in, pd_z_in, "gray")
        ax2.plot3D(pd_x_ex, pd_y_ex, pd_z_ex, "--")

    else:

        if args.data_type == "pendulum":
            gt_yline = 10 * np.sin(gt_xline)
            gt_xline = np.arange(0, 20, 0.1)
            pd_y_in = 10 * np.sin(pd_x_in)
            pd_y_ex = 10 * np.sin(pd_x_ex)
            pd_x_in, pd_x_ex = train_test_split(gt_xline, shuffle=False, test_size=0.8)

        ax0 = fig.add_subplot(131)
        ax1 = fig.add_subplot(132)
        ax2 = fig.add_subplot(133)

        ax0.plot(gt_xline, gt_yline)
        ax1.plot(pd_x_in, pd_y_in)
        ax1.plot(pd_x_ex, pd_y_ex, "red")
        ax2.plot(gt_xline, gt_yline)
        ax2.plot(pd_x_in, pd_y_in, "gray")
        ax2.plot(pd_x_ex, pd_y_ex, "--")

        print(pd_x_ex.shape, pd_y_ex.shape)

    graph_name = (
        "/data_seoul/mjeongp/SINDy_MASK/graph/"
        + args.data_type
        + "_"
        + str(mask_type)
        + "_err_"
        + str(loss.sum().item())
        + ".png"
    )
    fig.savefig(graph_name)

    if args.wandb:
        wandb.log({"Recon graph": [wandb.Image(graph_name)]})
