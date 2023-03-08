# SpReME

<p align="center">
  <img width="660" height="330" src="https://user-images.githubusercontent.com/46705780/223694737-7517a468-2061-4e26-b84b-5082ae79f9ba.png">
</p>

This repository contains official code for [SpReME: Sparse Regression for Multi-Environment Dynamic Systems](https://arxiv.org/pdf/2302.05942.pdf).

## How to Run Code

The training code finds a binary mask which represents shared sparse structure across the different environments.
You can run the training code with the script below.

```bash
python main.py --data-type <DATASET> --env-num <number of environment> --traj-num <number of trajectory per environment> --time <time horizon> --dt <time interval> --degree <polynomial degree>
```

- `-data-type`: dataset name (`linear`, `lorenz`, `lotka`, `pendulum`)
- `-degree`: maximum polynomial degree of candidate term (default value: 5)

You can also use arguments below.
- `--env-var`: variance of distribution which environment's parameter sampled from
- `--epoch`: epoch for training process
- `--mask-lr`: learning rate for mask training
- `--coef-lr`: learning rate for coefficient training
- `--mask-threshold`: threshold used when pruning elements of mask matrix
- `--coef-threshold`: threshold used when pruning elements of coefficeint matrix
- `--reg`: regularization weight for loss
- `--mask-schedule`: decaying rate of regularization weight for mask loss
- `--mask-timediv`: number of mask time horizon chunks

After you got trained mask, you can find adapted coefficient on unobserved environment with the mask with code below.

```bash
python adaptation.py --data-type <DATASET> --traj-num <number of trajectory per environment> --time <time horizon> --dt <time interval> --degree <polynomial degree>
```

- `-data-type`: dataset name (`linear`, `lorenz`, `lotka`, `pendulum`)
- `-degree`: maximum polynomial degree of candidate term (default value: 5) / This value should be same as train time.

The code contains not only adaptation but also test process.
The time horizon and dt value of test data is fixed to 2.5 times and 0.5 times of adaptation data, respectively.
