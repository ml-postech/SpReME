import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
import numpy as np
from sklearn import linear_model

from torch.utils.data import DataLoader

from generate_data import gen_data, PHY_dataset, cal_abs_max

from sklearn.preprocessing import PolynomialFeatures
from pysindy.feature_library import CustomLibrary
from pysindy.differentiation import SmoothedFiniteDifference
import copy

from functions import (
    print_mask_info,
    update_coef,
    update_mask,
    print_RMSE,
    cal_err,
    cal_mask_perf,
    args_set,
    init_by_SINDY,
    init_mask,
)

import matplotlib
import matplotlib.pyplot as plt


# ignore user warnings
import warnings
import random
import argparse


def main():
    warnings.filterwarnings("ignore", category=UserWarning)

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    wandb = bool(args.wandb)

    EPOCHS = args.epoch
    MASK_LR = args.mask_lr
    COEF_LR = args.coef_lr
    mask_threshold = args.mask_threshold
    coef_threshold = args.coef_threshold
    reg = args.reg
    mask_schedule = args.mask_schedule


    env_num = args.env_num
    traj_num = args.traj_num
    time = args.time
    dt = args.dt
    env_var = args.env_var
    time_chunk = args.mask_timediv
    data_type = args.data_type
    degree = args.degree

    train_data_param = {'data_type': data_type, 'time': time, 'dt':dt, 'traj_num':traj_num, 'env_num':env_num, 'env_var':env_var, 'degree':degree, 'seed': seed}   
    params, x0_trains, m, n, train_X, train_T = gen_data(**train_data_param)

    ### compute approximate derivative ###
    sfd = SmoothedFiniteDifference(smoother_kws={'window_length': 5})
    dot_X = []
    temp = train_X.transpose(1,0).cpu().numpy()
    for train in temp:
        trj_dot = []
        for trj in train:
            approx_dot = sfd._differentiate(trj, train_T[0].cpu().numpy())
            trj_dot.append(approx_dot)
        dot_X.append(np.array(trj_dot))
    
    dot_X = torch.tensor(np.array(dot_X)).transpose(1,0)
    ##########################################


    ##########################################################
    #############coef, mask 정의 및 initialization#############
    ##########################################################
    sparsity = 1.0

    if wandb:
        wandb.init(project="check")
        wandb.config.update(args)

    val_len = train_X.shape[2] // 5

    val_X = torch.tensor(train_X[:, :, -val_len-1:])
    val_T = torch.tensor(train_T[:, -val_len-1:])

    train_X = train_X[:, :, :-val_len]
    dot_X = dot_X[:, :, :-val_len]
    train_T = train_T[:, :-val_len]

    ############# divide mask time horizon ###################
    if time_chunk != 1:
        len = train_X.shape[2]
        len_want = len//int(traj_num/time_chunk)

        data_split = []
        T_split = []
        dot_split = []
        for i in range(0, len, len_want):
            data_split.append(train_X[:,:,i:i+len_want,:])
            T_split.append(train_T[:,i:i+len_want])
            dot_split.append(dot_X[:,:,i:i+len_want,:])
    
        train_X = torch.cat(data_split, dim=0)
        train_T = torch.cat(T_split, dim=0)
        dot_X = torch.cat(dot_split, dim=0)
    ##########################################################

    train_X = train_X.to(device)
    train_T = train_T.to(device)

    val_X = val_X.to(device)
    val_T = val_T.to(device)

    ################## get p information #####################################
    abs_max, _ = cal_abs_max(data_type, degree, train_X.cpu())
    p = abs_max.shape[-1]
    print("number of p : ",p)
    env_num = abs_max.shape[0] 
    ##########################################################################

    coef = init_by_SINDY(data_type, degree, dt, train_X, train_T[0], val_X, val_T[0], abs_max, dot_X)
    coef = coef.float().to(device)
    coef.requires_grad_(True)

    train_dataset = PHY_dataset(train_X, train_T)
    train_dataloader = DataLoader(train_dataset, batch_size=traj_num, shuffle=True)
    val_dataset = PHY_dataset(val_X, val_T)
    val_dataloader = DataLoader(val_dataset, batch_size=traj_num, shuffle=True)

    ##########################################################
    ##################### Fit the model ######################
    ##########################################################
    
    mask = nn.Parameter(init_mask(coef))
    precision, recall = cal_mask_perf(data_type, mask_threshold, torch.sigmoid(mask))
    print("After initialized: precision={}, recall={}".format(precision, recall))
    mask = mask.to(device)

    optimizer = optim.Adam([coef], lr=COEF_LR)
    mask_optimizer = optim.Adam([mask], lr=MASK_LR)
#######################
    min_err = float("inf")
    best_coef = None
    best_mask = None

    abs_max2=torch.tensor(np.expand_dims(abs_max,axis=2)).to(device)
    
    mask_val = torch.tensor(0.)
    err_per_env = torch.tensor(0.)
    
    for epoch in range(0, EPOCHS):
        for batch_X, batch_T in train_dataloader:

            if epoch % 50 == 1:
                print_mask_info(mask_threshold, torch.sigmoid(mask))
            
            mask_for_ct = copy.deepcopy(mask).detach()
            
            coef, cost_w_reg = update_coef(mask_threshold, degree, coef, mask_for_ct, batch_X, batch_T[0], optimizer, abs_max)    
            coef.data = torch.where(torch.abs(coef/abs_max2) < coef_threshold, 0.0, coef)
            print_RMSE(mask_threshold, degree, epoch, coef, mask, batch_X, batch_T[0], abs_max)
            
            coef_for_mt = copy.deepcopy(coef).detach()

            zeros = torch.zeros_like(mask)
            for env in range(env_num):
                zeros = torch.logical_or(zeros, coef[env]).float().detach()
            
            mask.data = torch.where(zeros == 0.0, -1e10, mask)
            mask.data = torch.where(torch.sigmoid(mask)<=mask_threshold, -1e10, mask)
            mask = update_mask(degree, mask_schedule, reg, env_num, coef_for_mt, mask, batch_X, batch_T[0], mask_optimizer, epoch, abs_max)
                        
            for (batch_val_X, batch_val_T) in val_dataloader:
                err_per_env = cal_err(mask_threshold, degree, coef, mask, batch_val_X, batch_val_T[0], abs_max)
                print(err_per_env.item())

            sparsity = ((torch.sigmoid(mask) <= mask_threshold).float().sum()) / (mask.shape[0] * mask.shape[1])

            if min_err >= err_per_env:
                min_err = err_per_env
                best_coef = copy.deepcopy(coef)
                best_mask = copy.deepcopy(mask)
                torch.save([(coef/abs_max2)*(torch.sigmoid(mask) > mask_threshold).float(), (torch.sigmoid(mask) > mask_threshold).float()], data_type+'_best.pt')

            precision, recall = cal_mask_perf(data_type, mask_threshold, torch.sigmoid(mask))
            
            if wandb:
                sparsity = ((torch.sigmoid(mask) <= mask_threshold).float().sum())/(mask.shape[0]*mask.shape[1])
                wandb.log(
                    {
                        "sparsity": sparsity,
                        "train loss": cost_w_reg / batch_X.shape[1],
                        "err wo reg": err_per_env,
                        "min_err wo reg": min_err,
                        "mask_val": mask_val,
                        "precision": precision,
                        "recall": recall,
                    }
                )
        if sparsity == 1.0 or recall != 1.0:
            break

    best_mask = (torch.sigmoid(best_mask) > mask_threshold).float()
    sparsity = 1-(best_mask.sum() / (best_mask.shape[0] * best_mask.shape[1]))
    precision, recall = cal_mask_perf(data_type, mask_threshold, best_mask)
    if wandb:
        wandb.config.update(
            {
                "best_mask": True,
                "b/f_sparsity": sparsity,
                "b/f_recall": recall,
                "b/f_precision": precision,
            }
        )
    print("*" * 50)
    print("*" * 50)
    print("")
    print(
        "Train Result\nparams=\n{}\nbest_mask=\n{}\nbest_coef=\n{}\nsparsity={},\tprecision={},\trecall={}".format(
            params,
            best_mask,
            best_coef,
            sparsity,
            precision,
            recall,
        )
    )
    print(coef*(torch.sigmoid(mask)>mask_threshold).float())

if __name__ == "__main__":

    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--seed",
        type=int,
        default=1000,
        metavar="S",
        help="random seed",
    )
    argparser.add_argument(
        "--data-type", type=str, default="lotka", help="data-type (2d or 3d or lorenz)"
    )
    argparser.add_argument("--traj-num", type=int, default=16, metavar="S", help="")
    argparser.add_argument("--test-traj", type=int, default=16, metavar="S", help="")
    argparser.add_argument("--env-num", type=int, default=3, metavar="S", help="")
    argparser.add_argument(
        "--env-var", type=float, help="variation of environment", default=0.01
    )
    argparser.add_argument("--test-env", type=int, default=1, metavar="S", help="")
    argparser.add_argument(
        "--degree", type=int, default=5, metavar="S", help="poly degree"
    )
    argparser.add_argument(
        "--dt", type=float, help="interval of train data", default=0.02
    )
    argparser.add_argument("--time", type=int, help="time length", default=0)
    argparser.add_argument(
        "--init-reg",
        type=float,
        help="regularization parameter when initializing with lasso",
        default=0,
    )
    argparser.add_argument("--epoch", type=int, help="epoch number", default=30000)
    argparser.add_argument(
        "--coef-epoch", type=int, help="number of coef update in a epoch", default=0
    )
    argparser.add_argument(
        "--mask-lr", type=float, help="mask_learning rate", default=0.01
    )
    argparser.add_argument(
        "--coef-lr", type=float, help="coef_learning rate", default=0.01
    )
    argparser.add_argument(
        "--mask-schedule", type=float, help="mask_learning rate", default=0.001
    )
    argparser.add_argument("--reg", type=float, help="regularization term", default=1e-3)
    argparser.add_argument(
        "--coef-threshold", type=float, help="threshold to making zero", default=1e-3
    )
    argparser.add_argument(
        "--mask-threshold", type=float, help="threshold to making zero", default=1e-5
    )
    argparser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
    argparser.add_argument(
        "--mask-timediv", type=int, help="divide mask time horizon n chunks", default=1
    )
    argparser.add_argument("--set-args", type=int, default=0, metavar="S", help="")
    argparser.add_argument("--wandb", type=int, default=0, metavar="S", help="")
    
    args = argparser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")

    if args.set_args == 0:
        pass
    else:
        args = args_set(args)

    if args.mask_type == 0:
        main()
    else:
        test(args, device)
