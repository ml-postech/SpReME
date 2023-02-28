import torch
import torch.nn as nn

import numpy as np

from pysindy.differentiation import SmoothedFiniteDifference
import argparse
from generate_data import PHY_dataset, cal_abs_max
import torch.optim as optim
from functions import predicted_func
from functions import predicted_func_clip, cal_err2, update_coef2

from torchdiffeq import odeint
from functools import partial
from torch.nn import functional as F
import copy



integrator_keywords = {}
integrator_keywords["method"] = "rk4"   

argparser = argparse.ArgumentParser()
argparser.add_argument(
    "--data-type", type=str, default="lorenz", help="data-type (2d or 3d or lorenz)")
argparser.add_argument(
    "--degree", type=int, default=5, metavar="S", help="poly degree")
argparser.add_argument(
    "--lr", type=float, default=0.01, metavar="S", help="learning rate")
argparser.add_argument(
    "--mt", type=int, default=0, metavar="S", help="learning rate")

def adaptation(mask, train_X, train_T):

    min_err = torch.tensor(float('inf'))
    best_coef = None
    for epoch in range(10000):
        loss = torch.tensor(0.).float().to('cuda')
        for i in range(train_T.shape[0]-1):
            x_train = odeint(
                    partial(
                        predicted_func_clip,
                        param= mask*coef.to('cuda').float(),
                        deg=args.degree,
                        abs_max = abs_max),
                    train_X[i:i+1],
                    train_T[i:i+2],
                    **integrator_keywords
                ).permute(1,0,2).squeeze()
            loss += F.smooth_l1_loss(x_train, train_X[i:i+2])

        loss /= train_T.shape[0]
        print(epoch, '] ',loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        pred = odeint(
                partial(
                    predicted_func_clip,
                    param= mask*coef.to('cuda').float(),
                    deg=args.degree,
                    abs_max = abs_max),
                valid_X[0:1],
                valid_T,
                **integrator_keywords
            ).permute(1,0,2).squeeze()

        err = F.mse_loss(pred, valid_X)
        if err < min_err:
            min_err = err
            best_coef = copy.deepcopy(coef)
            torch.save(best_coef, mask_type+'_coef_'+data+'.pt')

def test_prediction(coef, degree = 5):     
    pred = odeint(
                partial(
                    predicted_func,
                    param= coef.to('cuda').float(),
                    deg= degree,
                    abs_max = 1
                ),
                test_X[:,0,:],
                test_T[0],
                **integrator_keywords
            ).permute(1,0,2).squeeze()

    loss = F.mse_loss(pred, test_X, reduction = 'none')
    loss_mean = F.mse_loss(pred, test_X)
    
    loss = loss.mean(dim=(1,2))
    loss_max = torch.max(loss)
    loss_min = torch.min(loss)

    print('mean: ', loss_mean)
    print('min: ', loss_min)
    print('max: ', loss_max)


args = argparser.parse_args()
data = args.data_type

if args.mt == 0:
    mask_type = 'our'
elif args.mt == 1:
    mask_type = 'inter'
elif args.mt == 2:
    mask_type = 'union'
 
mask = None

if data == 'lorenz':
    mask = torch.zeros(56, 3).to('cuda').float()
    
    mask[1,0] = 1.0
    mask[2,0] = 1.0
    mask[1,1] = 1.0
    mask[2,1] = 1.0
    mask[6,1] = 1.0
    mask[8,1] = 1.0
    mask[0,2] = 1.0
    mask[3,2] = 1.0
    mask[4,2] = 1.0
    mask[5,2] = 1.0
    mask[7,2] = 1.0
    mask[6,2] = 1.0
    
    inter_mask = torch.tensor(
        [[False, False, False],
        [ True,  True, False],
        [ True,  True, False],
        [False, False,  True],
        [False, False,  True],
        [False, False,  True],
        [False, False, False],
        [False, False,  True],
        [False, False, False],
        [False, False, False],
        [False, False, False],
        [False, False, False],
        [False, False, False],
        [False, False, False],
        [False, False, False],
        [False, False, False],
        [False, False, False],
        [False, False, False],
        [False, False, False],
        [False, False, False],
        [False, False, False],
        [False, False, False],
        [False, False, False],
        [False, False, False],
        [False, False, False],
        [False, False, False],
        [False, False, False],
        [False, False, False],
        [False, False, False],
        [False, False, False],
        [False, False, False],
        [False, False, False],
        [False, False, False],
        [False, False, False],
        [False, False, False],
        [False, False, False],
        [False, False, False],
        [False, False, False],
        [False, False, False],
        [False, False, False],
        [False, False, False],
        [False, False, False],
        [False, False, False],
        [False, False, False],
        [False, False, False],
        [False, False, False],
        [False, False, False],
        [False, False, False],
        [False, False, False],
        [False, False, False],
        [False, False, False],
        [False, False, False],
        [False, False, False],
        [False, False, False],
        [False, False, False],
        [False, False, False]]).to('cuda').float()

    union_mask = torch.tensor(
       [[ True,  True,  True],
        [ True,  True,  True],
        [ True,  True,  True],
        [ True, False,  True],
        [ True, False,  True],
        [ True, False,  True],
        [ True,  True, False],
        [False, False,  True],
        [ True,  True, False],
        [False, False,  True],
        [False,  True, False],
        [False,  True, False],
        [False, False,  True],
        [False,  True, False],
        [False, False,  True],
        [False,  True, False],
        [False, False, False],
        [False, False, False],
        [False, False, False],
        [False, False, False],
        [False, False, False],
        [False, False, False],
        [False, False, False],
        [False, False, False],
        [False, False, False],
        [False, False, False],
        [False, False, False],
        [False, False, False],
        [False, False, False],
        [False, False, False],
        [False, False, False],
        [False, False, False],
        [False, False, False],
        [False, False, False],
        [False, False, False],
        [False, False, False],
        [False, False, False],
        [False, False, False],
        [False, False, False],
        [False, False, False],
        [False, False, False],
        [False, False, False],
        [False, False, False],
        [False, False, False],
        [False, False, False],
        [False, False, False],
        [False, False, False],
        [False, False, False],
        [False, False, False],
        [False, False, False],
        [False, False, False],
        [False, False, False],
        [False, False, False],
        [False, False, False],
        [False, False, False],
        [False, False, False]]).to('cuda')

    
elif data == 'pendulum':
    mask = torch.zeros(24, 2).to('cuda').float()
    
    mask[2, 0] = 1.0
    mask[2, 1] = 1.0
    mask[-3, 1] = 1.0
    
    union_mask = torch.tensor(
        [   [1, 1],
            [0, 1],
            [1, 1],
            [0, 1],
            [1, 0],
            [1, 1],
            [0, 1],
            [1, 1],
            [1, 1],
            [1, 1],
            [0, 1],
            [1, 1],
            [1, 1],
            [1, 0],
            [0, 0],
            [0, 1],
            [0, 1],
            [0, 1],
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
            [1, 1],
            [1, 1]]).to('cuda')
    inter_mask = torch.tensor(
        [   [0, 0],
            [0, 1],
            [1, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 1],
            [0, 1],
            [1, 1],
            [0, 0],
            [0, 0],
            [1, 1],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 1],
            [1, 0],
            [0, 1]]).to('cuda')

elif data == 'lotka':
    mask = torch.tensor(
        [[0., 0.],
        [1., 0.],
        [0., 1.],
        [0., 0.],
        [1., 1.],
        [0., 0.],
        [0., 0.],
        [0., 0.],
        [0., 0.],
        [0., 0.],
        [0., 0.],
        [0., 0.],
        [0., 0.],
        [0., 0.],
        [0., 0.],
        [0., 0.],
        [0., 0.],
        [0., 0.],
        [0., 0.],
        [0., 0.],
        [0., 0.]]).to('cuda')
    
    union_mask = torch.tensor(
        [[True, True],
        [True, True],
        [True, True],
        [True, True],
        [True, True],
        [True, True],
        [True, True],
        [True, True],
        [True, True],
        [True, True],
        [True, True],
        [True, True],
        [True, True],
        [True, True],
        [True, True],
        [True, True],
        [True, True],
        [True, True],
        [True, True],
        [True, True],
        [True, True]]).to('cuda').float()

    inter_mask = torch.tensor(
       [[ True,  True],
        [ True,  True],
        [ True,  True],
        [False, False],
        [ True,  True],
        [False,  True],
        [False, False],
        [ True,  True],
        [ True,  True],
        [False, False],
        [False, False],
        [False, False],
        [False, False],
        [False, False],
        [False, False],
        [False, False],
        [False, False],
        [False, False],
        [False, False],
        [False, False],
        [False, False]]).to('cuda').float()
elif data == '3d':
    mask = torch.zeros(56, 3).to('cuda').float()
    mask[1, 0] = 1.0
    mask[2, 0] = 1.0
    mask[1, 1] = 1.0
    mask[2, 1] = 1.0
    mask[3, 2] = 1.0

    union_mask = torch.tensor(
       [[False, False, False],
        [ True,  True, False],
        [ True,  True, False],
        [False, False,  True],
        [False, False, False],
        [False, False, False],
        [False,  True, False],
        [False, False, False],
        [ True, False, False],
        [False, False, False],
        [False,  True, False],
        [ True,  True, False],
        [False,  True, False],
        [ True,  True, False],
        [ True,  True, False],
        [False,  True, False],
        [ True, False, False],
        [ True, False, False],
        [ True, False, False],
        [False, False, False],
        [False, False, False],
        [False, False, False],
        [False,  True, False],
        [False, False, False],
        [ True,  True, False],
        [False, False, False],
        [False, False, False],
        [False,  True, False],
        [False, False, False],
        [False,  True, False],
        [False, False, False],
        [ True, False, False],
        [False, False, False],
        [ True, False, False],
        [False, False, False],
        [False,  True, False],
        [ True,  True, False],
        [False,  True, False],
        [ True,  True, False],
        [False,  True, False],
        [False,  True, False],
        [ True, False, False],
        [False,  True, False],
        [ True, False, False],
        [False,  True, False],
        [ True,  True, False],
        [False, False, False],
        [False,  True, False],
        [False, False, False],
        [False, False, False],
        [ True, False, False],
        [ True, False, False],
        [ True, False, False],
        [ True, False, False],
        [ True, False, False],
        [False, False, False]]).to('cuda').float()

    inter_mask = torch.tensor(
       [[False, False, False],
        [False,  True, False],
        [ True,  True, False],
        [False, False,  True],
        [False, False, False],
        [False, False, False],
        [False, False, False],
        [False, False, False],
        [False, False, False],
        [False, False, False],
        [False, False, False],
        [False, False, False],
        [False, False, False],
        [False, False, False],
        [False, False, False],
        [False, False, False],
        [False, False, False],
        [False, False, False],
        [False, False, False],
        [False, False, False],
        [False, False, False],
        [False, False, False],
        [False, False, False],
        [False, False, False],
        [False, False, False],
        [False, False, False],
        [False, False, False],
        [False, False, False],
        [False, False, False],
        [False, False, False],
        [False, False, False],
        [False, False, False],
        [False, False, False],
        [False, False, False],
        [False, False, False],
        [False, False, False],
        [False, False, False],
        [False, False, False],
        [False, False, False],
        [False, False, False],
        [False, False, False],
        [False, False, False],
        [False, False, False],
        [False, False, False],
        [False, False, False],
        [False, False, False],
        [False, False, False],
        [False, False, False],
        [False, False, False],
        [False, False, False],
        [False, False, False],
        [False, False, False],
        [False, False, False],
        [False, False, False],
        [False, False, False],
        [False, False, False]]).to('cuda').float()

if mask_type == 'union':
    mask = union_mask
elif mask_type == 'inter':
    mask = inter_mask

adaptation_train_param = {'data_type': data_type, 'time': time, 'dt': dt, 'traj_num':1, 
                    'env_num':1, 'env_var':0., 'seed': 100, 'train_time':True, 'adaptation':True}
adaptation_test_param = {'data_type': data_type, 'time': time, 'dt': dt, 'traj_num':1, 
                    'env_num':1, 'env_var':0., 'seed': 100, 'train_time':False, 'adaptation':True}


_, _, _, _, train_X, train_T = gen_data(**adaptation_train_param)
_, _, _, _, test_X, test_T = gen_data(**adaptation_test_param)


val_len = train_X.shape[2] // 5

val_X = torch.tensor(train_X[:, :, -val_len-1:])
val_T = torch.tensor(train_T[:, -val_len-1:])

train_X = train_X[:, :, :-val_len]
train_T = train_T[:, :-val_len]

abs_max, _ = cal_abs_max(args, train_X)

coef = nn.Parameter(torch.randn(mask.shape).float().to('cuda')*mask)
optimizer = optim.Adam([coef], lr=args.lr)

train_X = train_X.squeeze().to('cuda')
train_T = train_T.squeeze().to('cuda')

test_X = test_X.squeeze().to('cuda')
test_T = test_T.squeeze().to('cuda')

val_X = val_X.squeeze().to('cuda')
val_T = val_T.squeeze().to('cuda')

print(train_X.shape, test_X.shape)