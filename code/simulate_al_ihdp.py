# Import libraries and set constants and parameters.
import argparse
import os
import time
import itertools
import warnings
import torch
import torch.nn as nn
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
from causal_models import CMGP
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri
import rpy2

grf = importr("grf")
stats = importr("stats")
numpy2ri.activate()

torch.set_default_tensor_type('torch.cuda.DoubleTensor')
warnings.filterwarnings("ignore")
SQRT_CONST = 1e-10


# Set parameters. Different options are indicated in comments.
parser = argparse.ArgumentParser()

parser.add_argument('--data_path', '-d', type=str,
                    default='../data/ihdp/missing_0.95/')
parser.add_argument('--exp_nb_start', type=int, default=0)
parser.add_argument('--exp_nb_end', type=int, default=500)
parser.add_argument('--query_nb', type=int, default=44)
parser.add_argument('--model_type', type=str, default="MLP") #DoublyRobust, MLP, GP, CF, CMGP
parser.add_argument('--multihead', action="store_true")
parser.add_argument('--learn_type', '-l', type=str, default="loss") #random, a_uncertainty, loss
parser.add_argument('--loss_factual', action="store_true") #if true, we add the factual term
parser.add_argument('--loss_distance', type=str, default="") #mmd, wass, empty string
parser.add_argument('--save_pehe', action="store_true") #true if set
parser.add_argument('--save_loss', action="store_true") #true if set
parser.add_argument('--batch_size_small', type=int, default=5)
parser.add_argument('--batch_size_large', type=int, default=31)
parser.add_argument('--batch_switch', type=int, default=32)
parser.add_argument('--result_path', '-r', type=str,
                    default='../results/mmd/')
parser.add_argument('--sigma', type=float, default=10000)

args = parser.parse_args()


# Functions needed to run the main script.
def seed_everything(seed):
    """Function to radomize the experiment.

    Args:
        seed (int): random seed used to randomize the experiment.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False_0401


def build_datasets(prefix, i, rand_seed=0):
    """Function to read data and create datasets for training.

    Args:
        prefix (str): path to the datasets to read.
        i (int): experiment number.
        rand_seed (int): random seed.
    Returns:
        list(pd.DataFrame): Initial X/a/y training set, X/a/y to query from,
                            and X/a/y test set. 
    """
    seed_everything(rand_seed)
    # Read files
    X_masked = pd.read_csv(prefix+"X_mask_all_"+str(i)+".csv")
    X = pd.read_csv(prefix+"X_all_"+str(i)+".csv")
    y = pd.read_csv(prefix+"y_all_"+str(i)+".csv")

    # Separate masked and the rest of X
    X_masked = X_masked.dropna()
    y_masked = y[y.index.isin(X_masked.index)]
    X = X.loc[~X.index.isin(X_masked.index)]
    y = y.loc[~y.index.isin(X_masked.index)]

    X_pool, X_test, y_pool, y_test = train_test_split(X, y, random_state=rand_seed)
    X_pool.reset_index(drop=True, inplace=True)
    y_pool.reset_index(drop=True, inplace=True)

    a_masked = X_masked[['momwhite']]
    a_pool = X_pool[['momwhite']]
    a_test = X_test[['momwhite']]

    # Remove unnecessary categories and add useful category
    X_masked.drop(columns=['momwhite'], inplace=True)
    X_pool.drop(columns=['momwhite'], inplace=True)
    X_test.drop(columns=['momwhite'], inplace=True)

    return X_masked, y_masked, a_masked, X_pool, y_pool, a_pool, X_test, y_test, a_test


def pehe_loss(model, X_val, y_val, causal_forest=False, drnn=False):
    """Function to calculate the PEHE value.

    Args:
        model: model trained for treatment effect estimation.
        X_val (pd.DataFrame): covariates dataset to test the model on.
        y_val (pd.DataFrame): outcome dataset to test the model on.
        causal_forest (bool): if true, a specific method is used for outcome predictions.
        drnn (bool): if true, a specific method is used for outcome predictions.
    Returns:
        float: PEHE loss. 
    """
    # Calculate ITEs based on the outcome predictions.
    if causal_forest:
        y_pred_diff = stats.predict(model, np.array(X_val.drop(columns='t')))
        y_pred_diff = np.array(y_pred_diff)

    else:
        X_val_0 = X_val.copy()
        X_val_0.at[:, 't'] = 0

        X_val_1 = X_val.copy()
        X_val_1.at[:, 't'] = 1

        if drnn:
            y_pred_0 = model(torch.Tensor(np.array(X_val_0))).cpu().detach().numpy()
            y_pred_1 = model(torch.Tensor(np.array(X_val_1))).cpu().detach().numpy()
        else:
            y_pred_0 = model.predict(X_val_0)
            y_pred_1 = model.predict(X_val_1)

        y_pred_diff = y_pred_1 - y_pred_0

    # With "true" ITEs, calculate PEHE loss.
    y_val_diff = list(y_val['mu1'] - y_val['mu0'])

    y_loss = [y_val_diff[i] - y_pred_diff[i] for i in range(len(y_val_diff))]
    y_loss = [l ** 2 for l in y_loss]

    return np.sum(y_loss)


def ate_loss(model, X_val, y_val, causal_forest=False, drnn=False):
    """Function to calculate the ATE value.

    Args:
        model: model trained for treatment effect estimation.
        X_val (pd.DataFrame): covariates dataset to test the model on.
        y_val (pd.DataFrame): outcome dataset to test the model on.
        causal_forest (bool): if true, a specific method is used for outcome predictions.
        drnn (bool): if true, a specific method is used for outcome predictions.
    Returns:
        float: ATE loss. 
    """
    # Calculate ITEs based on the outcome predictions.
    if causal_forest:
        y_pred_diff = stats.predict(model, np.array(X_val.drop(columns='t')))
        y_pred_diff = np.array(list(y_pred_diff))

    else:
        X_val_0 = X_val.copy()
        X_val_0.at[:, 't'] = 0

        X_val_1 = X_val.copy()
        X_val_1.at[:, 't'] = 1

        if drnn:
            y_pred_0 = model(torch.Tensor(np.array(X_val_0))).cpu().detach().numpy()
            y_pred_1 = model(torch.Tensor(np.array(X_val_1))).cpu().detach().numpy()

        else:
            y_pred_0 = model.predict(X_val_0)
            y_pred_1 = model.predict(X_val_1)

        y_pred_diff = y_pred_1 - y_pred_0

    # With "true" ITEs, calculate ATE loss.
    y_val_diff = list(y_val['mu1'] - y_val['mu0'])

    return np.sum(y_val_diff) - np.sum(y_pred_diff)


def pehe_loss_multi(model_c, model_t, X_val_c, y_val_c, X_val_t, y_val_t):
    """Function to calculate the PEHE value for a multi-head model.

    Args:
        model: multi-head model trained for treatment effect estimation.
        X_val (pd.DataFrame): covariates dataset to test the model on.
        y_val (pd.DataFrame): outcome dataset to test the model on.
        causal_forest (bool): if true, a specific method is used for outcome predictions.
        drnn (bool): if true, a specific method is used for outcome predictions.
    Returns:
        float: PEHE loss. 
    """
    # Calculate ITEs based on the outcome predictions.
    y_pred_f_c = model_c.predict(X_val_c)
    y_pred_f_t = model_t.predict(X_val_t)

    X_val_cf_c = X_val_t.copy()
    X_val_cf_c.at[:, 't'] = 0
    X_val_cf_t = X_val_c.copy()
    X_val_cf_t.at[:, 't'] = 1
    y_pred_cf_c = model_c.predict(X_val_cf_c)
    y_pred_cf_t = model_t.predict(X_val_cf_t)

    y_pred_c = np.concatenate((y_pred_f_c, y_pred_cf_c), axis=None)
    y_pred_t = np.concatenate((y_pred_cf_t, y_pred_f_t), axis=None)
    y_pred_diff = y_pred_t - y_pred_c

    # With "true" ITEs, calculate PEHE loss.
    y_val = y_val_c.append(y_val_t)
    y_val_diff = list(y_val['mu1'] - y_val['mu0'])

    y_loss = y_val_diff - y_pred_diff
    y_loss = [l ** 2 for l in y_loss]

    return np.sum(y_loss)


def ate_loss_multi(model_c, model_t, X_val_c, y_val_c, X_val_t, y_val_t):
    """Function to calculate the ATE value for a multi-head model.

    Args:
        model: multi-head model trained for treatment effect estimation.
        X_val (pd.DataFrame): covariates dataset to test the model on.
        y_val (pd.DataFrame): outcome dataset to test the model on.
        causal_forest (bool): if true, a specific method is used for outcome predictions.
        drnn (bool): if true, a specific method is used for outcome predictions.
    Returns:
        float: ATE loss. 
    """
    y_pred_f_c = model_c.predict(X_val_c)
    y_pred_f_t = model_t.predict(X_val_t)

    X_val_cf_c = X_val_t.copy()
    X_val_cf_c.at[:, 't'] = 0
    X_val_cf_t = X_val_c.copy()
    X_val_cf_t.at[:, 't'] = 1
    y_pred_cf_c = model_c.predict(X_val_cf_c)
    y_pred_cf_t = model_t.predict(X_val_cf_t)

    y_pred_c = np.concatenate((y_pred_f_c, y_pred_cf_c), axis=None)
    y_pred_t = np.concatenate((y_pred_cf_t, y_pred_f_t), axis=None)
    y_pred_diff = y_pred_t - y_pred_c

    # With "true" ITEs, calculate ATE loss.
    y_val = y_val_c.append(y_val_t)
    y_val_diff = list(y_val['mu1'] - y_val['mu0'])

    return np.sum(y_val_diff) - np.sum(y_pred_diff)


def ce_loss(predictions, targets, epsilon=1e-12):
    """Function to calculate the cross-entropy loss given predictions and ground truth.

    Args:
        predictions (np.array): outcome values predicted by the model.
        targets (np.array): ground-truth outcome values.
        epsilon (float): value to clip the prediction values.
    Returns:
        float: cross-entropy loss. 
    """
    predictions = np.clip(predictions, epsilon, 1. - epsilon)
    N = predictions.shape[0]
    ce = -np.sum(targets*np.log(predictions+1e-9))/N
    return ce


def compute_uncertainty(model, X_pool, batch_size):
    """Function to compute the uncertainty given a model and query data 
    (baseline method). Finds samples with highest uncertainty of A.

    Args:
        model: model trained for estimating A.
        X_pool (pd.DataFrame): dataset of samples with missing information on A.
        batch_size (int): number of samples to return.
    Returns:
        np.array(int): indices of samples with highest uncertainty of A. 
    """
    predict = model.predict_proba(X_pool)
    uncertainty = 1 - predict.max(axis=1)
    return np.argsort(uncertainty)[-batch_size:]


def norm_sq(x):
    """Function to compute the norm of x.

    Args:
        x: numerical quantity (vector, array, torch.tensor, etc.).
    Returns:
        float: norm of x. 
    """
  return (x**2).sum()


def distance(x,y):
    """Function to compute the distance between two tensors.

    Args:
        x (torch.tensor): tensor 1.
        y (torch.tensor): tensor 2.
    Returns:
        float: distance value.
    """
    new_size = [x.size()[0]] + list(y.size())
    d = x.unsqueeze(1).expand(*new_size) - y.unsqueeze(0).expand(*new_size)
    return d.pow(2)


def mmd_rbf(x, y, sigma):
    """Function to compute mmdwith a RBF kernel.

    Args:
        x (torch.tensor): tensor 1.
        y (torch.tensor): tensor 2.
        sigma (float): value to set gamma.
    Returns:
        torch.tensor: mmd value.
    """
    gamma = 1/(2*(sigma**2))

    Kxx = (-gamma * distance(x,x).sum(dim=2)).exp()
    kyy = (-gamma * distance(y,y).sum(dim=2)).exp()
    kxy = (-gamma * distance(x,y).sum(dim=2)).exp()

    return torch.mean(Kxx) + torch.mean(kyy) - 2*torch.mean(kxy)


def mmd_fourier(x, y, sigma, dim_r=1024):
    """Function to compute the mmd using an approximate RBF kernel 
    by random features.
    
    Args:
        x (torch.tensor): tensor 1.
        y (torch.tensor): tensor 2.
        sigma (float): value to set gamma.
        dim_r (int): dimension to use for approximation.
    Returns:
        torch.tensor, torch.tensor: mmd values (biased and unbiased).
    """
    rnd_a = torch.empty((x1.size()[1], dim_r)).normal_()
    rnd_b = torch.empty(dim_r).uniform_()

    rW_n = 1/sigma * rnd_a
    rb_u = 2 * math.pi * rnd_b
    rW_n = rW_n.type(torch.cuda.DoubleTensor)
    rf0 = math.sqrt(2/dim_r) * torch.cos(x1.mm(rW_n) + rb_u.expand(x1.size()[0], dim_r).to(torch.float64))
    rf1 = math.sqrt(2/dim_r) * torch.cos(x2.mm(rW_n) + rb_u.expand(x2.size()[0], dim_r).to(torch.float64))

    k0=1
    nPos = x.size(0)
    nNeg = y.size(0)
    # Biased, unbiased
    return norm_sq(rf0.mean(0) - rf1.mean(0)), norm_sq(rf0.mean(0) - rf1.mean(0)) +  norm_sq(rf0.mean(0))/(nPos-1) + norm_sq(rf1.mean(0))/(nNeg-1) - k0*(nPos+nNeg-2)/(nPos-1)/(nNeg-1)


def pdist2sq(X,Y):
    """ Computes the squared Euclidean distance between all pairs x in X, y in Y
    
    Args:
        X (torch.tensor): tensor 1.
        Y (torch.tensor): tensor 2.
    Returns:
        torch.tensor: distance value.
    """
    C = -2*torch.matmul(X,torch.transpose(Y, 0, 1))
    nx = torch.sum(X**2, 1, keepdim=True)
    ny = torch.sum(Y**2, 1, keepdim=True)
    D = (C + torch.transpose(ny, 0, 1)) + nx
    return D


def safe_sqrt(x, lbound=SQRT_CONST):
    """Numerically stable version of sqrt
    Args:
        x (torch.tensor): tensor.
        lbound (float): min value to clip x to.
    Returns:
        torch.tensor: square root value.
    """
    return torch.sqrt(torch.clamp(x, lbound, np.inf))


def wass_distance(x, y, lam=10, its=10):
    """Function to compute the Wasserstein distance between x and y.
    
    Args:
        x (torch.tensor): tensor 1.
        y (torch.tensor): tensor 2.
        lam (float): lambda value.
        its (int): number of iterations.
    Returns:
        torch.tensor: Wasserstein distance value.
    """
    n1 = x.shape[0]
    n2 = y.shape[0]
    p = n1 / (n1 + n2)
    M = safe_sqrt(pdist2sq(x, Y=))

    # Estimate lambda and delta
    M_mean = torch.mean(M)
    delta = torch.max(M)
    eff_lam = lam/M_mean

    # Compute new distance matrix
    Mt = M
    row = delta*torch.ones(M[0:1,:].shape)
    col = torch.cat([delta*torch.ones(M[:,0:1].shape),torch.zeros((1,1))],0)
    Mt = torch.cat([M,row],0)
    Mt = torch.cat([Mt,col],1)

    # Compute marginal vectors
    a = torch.cat([p*torch.ones((n1, 1))/n1, (1-p)*torch.ones((1,1))],0)
    b = torch.cat([(1-p)*torch.ones((n2, 1))/n2, p*torch.ones((1,1))],0)

    # Compute kernel matrix
    Mlam = eff_lam*Mt
    K = torch.exp(-Mlam) + 1e-6 # added constant to avoid nan
    ainvK = K/a

    u = a
    for i in range(0,its):
        u = 1.0/(torch.matmul(ainvK,(b/torch.transpose(torch.matmul(torch.transpose(u, 0, 1),K), 0, 1))))
    v = b/(torch.transpose(torch.matmul(torch.transpose(u, 0, 1),K), 0, 1))

    T = u*(torch.transpose(v, 0, 1)*K)
    E = T*Mt
    wass_dist = 2*torch.sum(E)

    return wass_dist


def create_cand_X(X_cand, momwhite):
    """Add momwhite value to X.
    Args:
        X (pd.DataFrame): covariates dataset.
        momwhite (int): value to set momwhite to.
    Returns:
        pd.DataFrame: dataset with momwhite value.
    """
    X_cand['momwhite'] = momwhite
    return X_cand


def create_cand_X_w_t(X_cand, momwhite, treatment):
    """Add momwhite and treatment values to X.
    Args:
        X (pd.DataFrame): covariates dataset.
        momwhite (int): value to set momwhite to.
        treatment (int): value to set treatment to.
    Returns:
        pd.DataFrame: dataset with momwhite and treatment values.
    """
    X_cand['momwhite'] = momwhite
    X_cand['t'] = treatment
    return X_cand


def dist_per_label(X_cand, treatment_cand, X_train_wa_t0, X_train_wa_t1,
                   dist_type="mmd", mmd_method="fourier"):
    """Calculate distance between control / treated populations when 
    adding X_cand.
    Args:
        X_cand (pd.DataFrame): candidate covariates.
        treatment_cand (int): treatment value.
        X_train_wa_t0 (pd.DataFrame): control population. 
        X_train_wa_t1 (pd.DataFrame): treated population.
        dist_type (str): distance type (choose between mmd and wass).
        mmd_method (str): mmd method (choose between fourier and rbf).
    Returns:
        float: distance between control / treated populations.
    """
    if treatment_cand == 0:
        tensor_0 = torch.tensor(X_train_wa_t0.append(X_cand).values)
        tensor_1 = torch.tensor(X_train_wa_t1.values)
    else:
        tensor_0 = torch.tensor(X_train_wa_t0.values)
        tensor_1 = torch.tensor(X_train_wa_t1.append(X_cand).values)

    if dist_type == "mmd":
        if mmd_method == "rbf":
            dist = mmd_rbf(tensor_0, tensor_1, args.sigma)
        else:
            dist = mmd_fourier(tensor_0, tensor_1, args.sigma)
    else:
        dist = wass_distance(tensor_0, tensor_1)

    return dist[0].item()


def dist_exploration(X_train, a_train, X_pool, a_pool, treatment,
                     dist_type="mmd", mmd_method="fourier"):
    """Calculate distance between the training population and the masked population
    (pool to query data from). Used for EDA.
    Args:
        X_train (pd.DataFrame): candidate covariates from training dataset.
        a_train (pd.DataFrame): A values from training dataset.
        X_train (pd.DataFrame): candidate covariates from pool to query from.
        a_train (pd.DataFrame): A values from pool to query from.
        treatment (int): treatment value.
        dist_type (str): distance type (choose between mmd and wass).
        mmd_method (str): mmd method (choose between fourier and rbf).
    Returns:
        float: distance between training / masked populations.
    """
    X_train_wa = pd.concat([X_train, a_train], axis=1)
    X_train_use = X_train_wa[X_train_wa['t']==treatment].drop(columns=['t'])
    X_pool_wa = pd.concat([X_pool, a_pool], axis=1)
    X_pool_use = X_pool_wa[X_pool_wa['t']==treatment].drop(columns=['t'])

    if min(X_train_use.shape[0], X_pool_use.shape[0]) < 2:
        return np.nan

    tensor_0 = torch.tensor(X_train_use.values)
    tensor_1 = torch.tensor(X_pool_use.values)

    if dist_type == "mmd":
        if mmd_method == "rbf":
            dist = mmd_rbf(tensor_0, tensor_1, args.sigma)
        else:
            dist = mmd_fourier(tensor_0, tensor_1, args.sigma)
    else:
        dist = wass_distance(tensor_0, tensor_1)

    return dist[0].item()


def compute_final_loss(model, X_train_wa_t0, X_train_wa_t1, X_pool, y_pool,
                       batch_size, j,
                       loss_distance, loss_factual, 
                       outcome_model_c=None, outcome_model_t=None, seed=0):
    """Calculate the final loss for a given iteration. The loss can be either distance-related (CB)
    or factual loss (OE).
    Args: 
        model: model used to predict the values of A.
        X_train_wa_t0 (pd.DataFrame): X train dataset with A, for control group.
        X_train_wa_t1 (pd.DataFrame): X train dataset with A, for treated group.
        X_pool (pd.DataFrame): X candidate dataset (data to query from).
        y_pool (pd.DataFrame): y candidate dataset (data to query from).
        batch_size (int): number of samples to query from.
        j (int): iteration number.
        loss_distance (str): type of distance (MMD or Wasserstein distance).
        loss_factual (bool): if set, calculate the outcome error loss.
        outcome_model_c: model used to predict the outcome for the control group. If the model
                         is not multi-head, this model is used as the overall model.
        outcome_model_t: model used to predict the outcome for the treated group.
        seed (int): random seed number.
    Returns:
        list(int), list(float),
        list(float): indexes and values of the minimum losses,
                     list of the mmd distances with these candidate observations.
    """
    seed_everything(seed)

    # Create predictions for A
    a_predict = model.predict_proba(X_pool)

    # Initialize loss list
    dist_list = []
    fact_list = []
    final_loss_list = []

    for i in range(len(X_pool)):
        # Go through each query point, and generate all possible candidates
        treatment_cand = int(X_pool['t'].iloc[i])
        X_cand = X_pool.drop(columns=['t']).iloc[i]
        X_cand_w_t = X_pool.iloc[i]
        X_cand_0 = create_cand_X(X_cand.copy(), 0)
        X_cand_1 = create_cand_X(X_cand.copy(), 1)
        X_cand_w_t_0 = create_cand_X(X_cand_w_t.copy(), 0)
        X_cand_w_t_1 = create_cand_X(X_cand_w_t.copy(), 1)

        # Compute 2 mmds
        if loss_distance != "":
            dist_0 = dist_per_label(X_cand_0, treatment_cand, X_train_wa_t0, X_train_wa_t1, distance_type) * a_predict[i][0]
            dist_1 = dist_per_label(X_cand_1, treatment_cand, X_train_wa_t0, X_train_wa_t1, distance_type) * a_predict[i][1]
            dist_list.append(dist_0 + dist_1)

        else:
            dist_0 = dist_1 = 0
            dist_list.append(0)

        if loss_factual:
            # Compute the outcome loss
            if args.multihead:
                if treatment_cand == 1:
                    pred_0 = outcome_model_t.predict(X_cand_w_t_0.values.reshape(1, -1)).item(0)
                    pred_1 = outcome_model_t.predict(X_cand_w_t_1.values.reshape(1, -1)).item(0)
                else:
                    pred_0 = outcome_model_c.predict(X_cand_w_t_0.values.reshape(1, -1)).item(0)
                    pred_1 = outcome_model_c.predict(X_cand_w_t_1.values.reshape(1, -1)).item(0)
            else:
                if args.model_type == "CMGP":
                    pred_full_0 = outcome_model_c.predict(X_cand_0)
                    pred_full_1 = outcome_model_c.predict(X_cand_1)
                    if treatment_cand == 1:
                        pred_0 = pred_full_0[2][0][0]
                        pred_1 = pred_full_1[2][0][0]
                    else:
                        pred_0 = pred_full_0[1][0][0]
                        pred_1 = pred_full_1[1][0][0]

                elif args.model_type == "DoublyRobust":
                    x0 = torch.Tensor(X_cand_w_t_0.values.reshape(1, -1))
                    x1 = torch.Tensor(X_cand_w_t_1.values.reshape(1, -1))
                    pred_0 = outcome_model_c(x0).cpu().detach().numpy().item(0)
                    pred_1 = outcome_model_c(x1).cpu().detach().numpy().item(0)

                elif args.model_type != "CF":
                    pred_0 = outcome_model_c.predict(X_cand_w_t_0.values.reshape(1, -1)).item(0)
                    pred_1 = outcome_model_c.predict(X_cand_w_t_1.values.reshape(1, -1)).item(0)

                else:
                    raise NameError('Causal Forest does not handle factual error!')

            pred = pred_0 * a_predict[i][0] + pred_1 * a_predict[i][1]
            pred_error = np.abs(y_pool['y_factual'].iloc[i] - pred)
            fact_list.append(pred_error)
        else:
            pred_0 = pred_1 = pred_error = 0
            fact_list.append(0)

        # We add the final loss
        final_dist = dist_0 + dist_1 if loss_distance != "" else 0
        final_factual = pred_error if (loss_factual) else 0
        final_loss_list.append(final_dist - final_factual)

        min_loss_idx = np.argsort(np.array(final_loss_list))[:batch_size]
        min_loss = np.sort(np.array(final_loss_list))[:batch_size]

        if args.save_loss:
            with open(args.result_path+"loss/"+ "experiment_" + str(seed) + "_all_losses.csv", "a+") as f:
                    f.write("%d,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.12f,%.12f\n"%(j+1, i, a_predict[i][0], a_predict[i][1], \
                                                            pred_0, pred_1, pred_error,
                                                            dist_0, dist_1))

    return min_loss_idx, min_loss, dist_list


def DoublyRobustNN(X, y, X_test, y_test, multihead, seed=0):
    """Train a doubly robust model with NN to predict outcome.
    Args:
        X (pd.DataFrame): X train dataset.
        y (pd.DataFrame): y train dataset.
        X_test (pd.DataFrame): X test dataset.
        y_test (pd.DataFrame): y test dataset.
        multihead (bool): indication if a multi-head model is used.
        seed (int): radom seed number.
    Returns:
        model, model, float, float: control / treated models (if not multihead, both will refer to
                                    the same single model), PEHE value, ATE value.
    """
    if multihead:
        raise NameError('Doubly Robust does not handle multihead!')
    seed_everything(seed)

    X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=X['t'], random_state=seed)
    #best_clf = LinearRegression()
    best_clf = nn.Sequential()
    best_pehe = float('Inf')

    # Define the loss
    X_train_1 = X_train.drop(columns=['t'])
    X_train_1 = torch.Tensor(np.array(X_train_1))
    y_train_1 = torch.Tensor(np.array(X_train['t']))
    X_val_1 = X_val.drop(columns=['t'])
    X_val_1 = torch.Tensor(np.array(X_val_1))
    y_val_1 = torch.Tensor(np.array(X_val['t']))
    criterion = nn.BCELoss()
    epochs = 300

    # Optimizers require the parameters to optimize and a learning rate
    alpha_list = [0.0005, 0.002, 0.005, 0.02, 0.05]

    best_clf_score = nn.Sequential()
    best_loss = float('Inf')
    for alp in alpha_list:
        model_1 = nn.Sequential(nn.Linear(X_train_1.shape[1], 10),
                            nn.ReLU(),
                            nn.Linear(10, 1),
                            nn.Sigmoid())
        optimizer = torch.optim.Adam(model_1.parameters(), lr=alp)
        model_1.train()
        for e in range(epochs):
            epoch_loss = 0
            optimizer.zero_grad()
            y_pred = model_1(X_train_1)
            loss = criterion(y_pred, y_train_1.unsqueeze(1))
            loss.backward()
            optimizer.step()
            epoch_loss = loss.item()
        loss_1 = criterion(model_1(X_val_1), y_val_1.unsqueeze(1)).item()
        if loss_1 < best_loss:
            best_clf_score = model_1
            best_loss = loss_1

    prop_score_train = best_clf_score(X_train_1)
    weights = 1/prop_score_train
    weights[weights>100] = 100

    def MSELossNN(output, target):
        return torch.sum(weights * (output - target) ** 2)

    criterion_2 = MSELossNN

    X_train_2 = torch.Tensor(np.array(X_train))
    y_train_2 = torch.Tensor(np.array(y_train['y_factual']))

    for alp in alpha_list:
        model_2 = nn.Sequential(nn.Linear(X_train_2.shape[1], 10),
                            nn.ReLU(),
                            nn.Linear(10, 1))
        optimizer = torch.optim.Adam(model_2.parameters(), lr=alp)
        model_2.train()
        for e in range(epochs):
            epoch_loss = 0
            optimizer.zero_grad()
            y_pred = model_2(X_train_2)
            loss = criterion_2(y_pred, y_train_2.unsqueeze(1))
            loss.backward(retain_graph=True)
            optimizer.step()
            epoch_loss = loss.item()
        pehe = pehe_loss(model_2, X_val, y_val, drnn=True)
        if pehe < best_pehe:
            best_clf = model_2
            best_pehe = pehe

    best_pehe = np.sqrt(pehe_loss(best_clf, X_test, y_test, drnn=True) / X_test.shape[0])
    best_ate = np.abs(ate_loss(best_clf, X_test, y_test, drnn=True) / X_test.shape[0])
    best_clf_c = best_clf
    best_clf_t = best_clf

    return best_clf_c, best_clf_t, best_pehe, best_ate


def DoublyRobust(X, y, X_test, y_test, multihead, seed=0):
    """Train a doubly robust model to predict outcome.
    Args:
        X (pd.DataFrame): X train dataset.
        y (pd.DataFrame): y train dataset.
        X_test (pd.DataFrame): X test dataset.
        y_test (pd.DataFrame): y test dataset.
        multihead (bool): indication if a multi-head model is used.
        seed (int): radom seed number.
    Returns:
        model, model, float, float: control / treated models (if not multihead, both will refer to
                                    the same single model), PEHE value, ATE value.
    """
    if multihead:
        raise NameError('Doubly Robust does not handle multihead!')
    seed_everything(seed)

    X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=X['t'], random_state=seed)
    best_clf = LinearRegression()
    best_pehe = float('Inf')

    penalty = ['l1', 'l2']
    C = [0.1, 1, 10, 100, 1000]
    hyperparameters_log = list(itertools.product(penalty, C))
    hyperparameters = ['True', 'False']

    for hp in hyperparameters:
        for hp_log in hyperparameters_log:
            best_clf_score, best_loss_score = None, float('Inf')
            clf_score = LogisticRegression(penalty=hp_log[0],
                                            C=hp_log[1],
                                            solver='liblinear')
            clf_score.fit(X_train.drop(columns=['t']), X_train['t'])
            clf_pred = clf_score.predict_proba(X_val.drop(columns=['t']))[:,1]
            loss_score = ce_loss(clf_pred, X_val['t'])
            if loss_score < best_loss_score:
                best_clf_score = clf_score
        prop_score_train = best_clf_score.predict_proba(X_train.drop(columns=['t']))[:,1]
        weights = 1/prop_score_train
        weights[weights > 100] = 100
        clf = LinearRegression(fit_intercept=hp)
        clf.fit(X_train, y_train['y_factual'], sample_weight=weights)
        pehe = pehe_loss(clf, X_val, y_val)
        if pehe < best_pehe:
            best_clf = clf
            best_pehe = pehe

    best_pehe = np.sqrt(pehe_loss(best_clf, X_test, y_test) / X_test.shape[0])
    best_ate = np.abs(ate_loss(best_clf, X_test, y_test) / X_test.shape[0])
    best_clf_c = best_clf
    best_clf_t = best_clf

    return best_clf_c, best_clf_t, best_pehe, best_ate


def MLP(X, y, X_test, y_test, multihead, seed=0):
    """Train a MLP model with NN to predict outcome.
    Args:
        X (pd.DataFrame): X train dataset.
        y (pd.DataFrame): y train dataset.
        X_test (pd.DataFrame): X test dataset.
        y_test (pd.DataFrame): y test dataset.
        multihead (bool): indication if a multi-head model is used.
        seed (int): radom seed number.
    Returns:
        model, model, float, float: control / treated models (if not multihead, both will refer to
                                    the same single model), PEHE value, ATE value.
    """
    seed_everything(seed)

    X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=X['t'], random_state=seed)
    best_pehe = float('Inf')

    X_train = X_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    X_val = X_val.reset_index(drop=True)
    y_val = y_val.reset_index(drop=True)

    # Create regularization hyperparameter space
    hid_lay = [(10, 10)]
    activ = ['relu']
    alp = [0.0005, 0.002, 0.005, 0.02, 0.05]
    learn_rate = ['constant', 'adaptive']
    hyperparameters = list(itertools.product(hid_lay, activ, alp, learn_rate))

    if multihead:
        best_clf_c = MLPRegressor()
        best_clf_t = MLPRegressor()
        X_train_c = X_train[X_train['t'] == 0] #.drop(columns=['t'])
        y_train_c = y_train[y_train.index.isin(X_train_c.index)]
        X_train_t = X_train.loc[~X_train.index.isin(X_train_c.index)]
        y_train_t = y_train.loc[~y_train.index.isin(X_train_c.index)]
        X_val_c = X_val[X_val['t'] == 0]
        y_val_c = y_val[y_val.index.isin(X_val_c.index)]
        X_val_t = X_val.loc[~X_val.index.isin(X_val_c.index)]
        y_val_t = y_val.loc[~y_val.index.isin(X_val_c.index)]
        X_test_c = X_test[X_test['t'] == 0]
        y_test_c = y_test[y_test.index.isin(X_test_c.index)]
        X_test_t = X_test.loc[~X_test.index.isin(X_test_c.index)]
        y_test_t = y_test.loc[~y_test.index.isin(X_test_c.index)]
        clf_c_list = []
        clf_t_list = []
        for hp in hyperparameters:
            clf_c = MLPRegressor(max_iter=300,
                                hidden_layer_sizes=hp[0],
                                activation=hp[1],
                                alpha=hp[2],
                                learning_rate=hp[3])
            clf_c.fit(X_train_c, y_train_c['y_factual'])
            clf_c_list.append(clf_c)
            clf_t = MLPRegressor(max_iter=300,
                                hidden_layer_sizes=hp[0],
                                activation=hp[1],
                                alpha=hp[2],
                                learning_rate=hp[3])
            clf_t.fit(X_train_t, y_train_t['y_factual'])
            clf_t_list.append(clf_t)
        for i in range(len(hyperparameters)):
            for j in range(len(hyperparameters)):
                pehe = np.sqrt(pehe_loss_multi(clf_c_list[i], clf_t_list[j], X_val_c, y_val_c, X_val_t, y_val_t) / X_val.shape[0])
                if pehe < best_pehe:
                    best_clf_t = clf_t_list[j]
                    best_clf_c = clf_c_list[i]
                    best_pehe = pehe
        best_pehe = np.sqrt(pehe_loss_multi(best_clf_c, best_clf_t, X_test_c, y_test_c, X_test_t, y_test_t) / X_test.shape[0])
        best_ate = np.abs(ate_loss_multi(best_clf_c, best_clf_t, X_test_c, y_test_c, X_test_t, y_test_t) / X_test.shape[0])
    else:
        best_clf = MLPRegressor()
        for hp in hyperparameters:
            clf = MLPRegressor(max_iter=300,
                                hidden_layer_sizes=hp[0],
                                activation=hp[1],
                                alpha=hp[2],
                                learning_rate=hp[3])
            clf.fit(X_train, y_train['y_factual'])
            pehe = np.sqrt(pehe_loss(clf, X_val, y_val) / X_val.shape[0])
            if pehe < best_pehe:
                best_clf = clf
                best_pehe = pehe
        best_pehe = np.sqrt(pehe_loss(best_clf, X_test, y_test) / X_test.shape[0])
        best_ate = np.abs(ate_loss(best_clf, X_test, y_test) / X_test.shape[0])
        best_clf_c = best_clf
        best_clf_t = best_clf
    return best_clf_c, best_clf_t, best_pehe, best_ate


def CF(X, y, X_test, y_test, multihead, seed=0):
    """Train a causal forest model with NN to predict outcome.
    Args:
        X (pd.DataFrame): X train dataset.
        y (pd.DataFrame): y train dataset.
        X_test (pd.DataFrame): X test dataset.
        y_test (pd.DataFrame): y test dataset.
        multihead (bool): indication if a multi-head model is used.
        seed (int): radom seed number.
    Returns:
        model, model, float, float: control / treated models (if not multihead, both will refer to
                                    the same single model), PEHE value, ATE value.
    """
    if multihead:
        raise NameError('Causal Forest does not handle multihead!')
    seed_everything(seed)
    X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=X['t'], random_state=seed)

    sample_frac = [0.3, 0.5]
    min_node_size = [3, 5]
    honesty = [True, False]
    hyperparameters = list(itertools.product(sample_frac, min_node_size, honesty))
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    best_clf = None
    best_pehe = float('Inf')
    for hp in hyperparameters:
        clf = grf.causal_forest(np.array(X_train.drop(columns='t')), \
                                    np.array(y_train['y_factual']).reshape(-1,1), \
                                    np.array(X_train['t']).reshape(-1,1),seed=0,
                                    **{'sample.fraction': hp[0],'min.node.size':hp[1],'honesty':hp[2], 'num.trees':500})
        pehe = np.sqrt(pehe_loss(clf, X_val, y_val, causal_forest=True) / X_val.shape[0])
        if pehe < best_pehe:
                best_clf = clf
                best_pehe = pehe
    best_pehe = np.sqrt(pehe_loss(best_clf, X_test, y_test, causal_forest=True) / X_test.shape[0])
    best_ate = np.abs(ate_loss(best_clf, X_test, y_test, causal_forest=True) / X_test.shape[0])

    return best_clf, best_clf, best_pehe, best_ate


def GP(X_train, y_train, X_test, y_test, multihead,
       outcome_model_c=None, outcome_model_t=None, kernel=None, seed=0):
    """Train a MLP model with NN to predict outcome.
    Args:
        X (pd.DataFrame): X train dataset.
        y (pd.DataFrame): y train dataset.
        X_test (pd.DataFrame): X test dataset.
        y_test (pd.DataFrame): y test dataset.
        multihead (bool): indication if a multi-head model is used.
        outcome_model_c: already trained outcome model for the control group, if multihead = True.
        outcome_model_t:already trained outcome model for the treated group, if multihead = True.
        kernel (str): kernel type for GP.
        seed (int): radom seed number.
    Returns:
        model, model, float, float: control / treated models (if not multihead, both will refer to
                                    the same single model), PEHE value, ATE value.
    """
    if outcome_model_c == None:
        outcome_model_c = GaussianProcessRegressor(kernel=kernel, alpha=0.01 ** 2, n_restarts_optimizer=10)
    if outcome_model_t == None:
        outcome_model_t = GaussianProcessRegressor(kernel=kernel, alpha=0.01 ** 2, n_restarts_optimizer=10)
    X_train = X_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    if multihead:
        X_train_c = X_train[X_train['t'] == 0] #.drop(columns=['t'])
        y_train_c = y_train[y_train.index.isin(X_train_c.index)]
        X_train_t = X_train.loc[~X_train.index.isin(X_train_c.index)]
        y_train_t = y_train.loc[~y_train.index.isin(X_train_c.index)]
        X_test_c = X_test[X_test['t'] == 0]
        y_test_c = y_test[y_test.index.isin(X_test_c.index)]
        X_test_t = X_test.loc[~X_test.index.isin(X_test_c.index)]
        y_test_t = y_test.loc[~y_test.index.isin(X_test_c.index)]
        outcome_model_c.fit(X_train_c, y_train_c['y_factual'])
        outcome_model_t.fit(X_train_t, y_train_t['y_factual'])
        best_pehe = np.sqrt(pehe_loss_multi(outcome_model_c, outcome_model_t, X_test_c, y_test_c, X_test_t, y_test_t) / X_test.shape[0])
        best_ate = np.abs(ate_loss_multi(outcome_model_c, outcome_model_t, X_test_c, y_test_c, X_test_t, y_test_t) / X_test.shape[0])

    else:
        outcome_model_c.fit(X_train, y_train['y_factual'])
        outcome_model_t = outcome_model_c
        best_pehe = np.sqrt(pehe_loss(outcome_model_c, X_test, y_test) / X_test.shape[0])
        best_ate = np.abs(ate_loss(outcome_model_c, X_test, y_test) / X_test.shape[0])

    return outcome_model_c, outcome_model_t, best_pehe, best_ate


def CausalMGP(X_train, y_train, X_test, y_test, multihead,
              outcome_model_c=None, outcome_model_t=None, seed=0):
    """Train a MLP model with NN to predict outcome.
    Args:
        X (pd.DataFrame): X train dataset.
        y (pd.DataFrame): y train dataset.
        X_test (pd.DataFrame): X test dataset.
        y_test (pd.DataFrame): y test dataset.
        multihead (bool): indication if a multi-head model is used.
        outcome_model_c: already trained outcome model for the control group, if multihead = True.
        outcome_model_t:already trained outcome model for the treated group, if multihead = True.
        seed (int): radom seed number.
    Returns:
        model, model, float, float: control / treated models (if not multihead, both will refer to
                                    the same single model), PEHE value, ATE value.
    """
    if multihead:
        raise NameError('Doubly Robust does not handle multihead!')
    seed_everything(seed)
    outcome_model_c = CMGP(dim=X_train.shape[1]-1, mode="CMGP")
    X_train = X_train.reset_index(drop=True).rename(columns={'t': 'W'})
    y_train = y_train.reset_index(drop=True).rename(columns={'y_factual': 'Y'})
    t_train = X_train[['W']]
    X_train_t = X_train.drop(columns=['W'])
    X_train_t.columns = range(0, 26)
    # X_train = X_train.rename(columns={25: 'W', 26: 25})

    X_test_t = X_test.reset_index(drop=True).drop(columns=['t'])
    X_test_t.columns = range(0, 26)
    outcome_model_c.fit(X_train_t, y_train[['Y']], t_train)
    outcome_model_t = outcome_model_c
    _, Y_est_0, Y_est_1 = outcome_model_c.predict(X_test_t)
    y_pred_diff = Y_est_1 - Y_est_0
    y_pred_diff = [i[0] for i in y_pred_diff]
    y_test_diff = list(y_test['mu1'] - y_test['mu0'])
    y_loss = [y1 - y2 for (y1, y2) in zip(y_test_diff, y_pred_diff)]
    y_loss = [l ** 2 for l in y_loss]
    best_pehe = np.sqrt(np.sum(y_loss) / X_test.shape[0])
    best_ate = np.abs(np.mean(y_test_diff) - np.mean(y_pred_diff))

    return outcome_model_c, outcome_model_t, best_pehe, best_ate


# MAIN
if __name__ == '__main__':
    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)
    if not os.path.exists(args.result_path + 'errors/'):
        os.makedirs(args.result_path + 'errors/')
    if not os.path.exists(args.result_path + 'loss/'):
        os.makedirs(args.result_path + 'loss/')

    model_type = args.model_type
    for i in range(args.exp_nb_start, args.exp_nb_end):
        seed_everything(i)

        start = time.time()
        X_train, y_train, a_train, X_pool, y_pool, a_pool, X_test, y_test, a_test = build_datasets(args.data_path, i, i)
        X_test_wa = pd.concat([X_test, a_test], axis=1)
        X_test_wa.to_csv(args.result_path+'X_test_'+args.learn_type+'_'+str(i)+'.csv', index=False)
        y_test.to_csv(args.result_path+'y_test_'+args.learn_type+'_'+str(i)+'.csv', index=False)
        X_train_wa = pd.concat([X_train, a_train], axis=1)
        X_train_wa_t0 = X_train_wa[X_train_wa['t'] == 0].drop(columns=['t'])
        X_train_wa_t1 = X_train_wa[X_train_wa['t'] == 1].drop(columns=['t'])

        # Initialize query and prediction model
        if model_type == "GP":
            kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(1, (1e-2, 1e2))
            outcome_model_c, outcome_model_t, final_pehe, final_ate = GP(X_train_wa, y_train, X_test_wa, y_test, args.multihead, None, None, kernel, i)
        elif model_type == "MLP":
            outcome_model_c, outcome_model_t, final_pehe, final_ate = MLP(X_train_wa, y_train, X_test_wa, y_test, args.multihead, i)
        elif model_type == "DoublyRobust":
            outcome_model_c, outcome_model_t, final_pehe, final_ate = DoublyRobustNN(X_train_wa, y_train, X_test_wa, y_test, args.multihead, i)
        elif model_type == "CF":
            outcome_model_c, outcome_model_t, final_pehe, final_ate = CF(X_train_wa, y_train, X_test_wa, y_test, args.multihead, i)
        elif model_type == "CMGP":
            outcome_model_c, outcome_model_t, final_pehe, final_ate = CausalMGP(X_train_wa, y_train, X_test_wa, y_test, args.multihead, i)
        # Record final results

        if args.save_pehe:
            dist_explore_t = dist_explore_c = 0
            dist_explore_t = dist_exploration(X_train, a_train, X_pool, a_pool, 1)
            dist_explore_c = dist_exploration(X_train, a_train, X_pool, a_pool, 0)
            with open(args.result_path+"errors/"+ "error.csv", "a+") as f:
                f.write("%s,%s,%s,%s,%d,%d,%d,%.4f,%.4f,%.12f,%.12f\n"%(model_type, args.learn_type, args.loss_distance, args.loss_factual,
                                                i,0,X_train.shape[0],final_pehe,final_ate, dist_explore_t, dist_explore_c))

        model = RandomForestClassifier(n_estimators=100)
        model.fit(X_train, a_train['momwhite'])

        for j in range(args.query_nb):
            if j < args.batch_switch:
                batch_size = args.batch_size_small
            else:
                batch_size = args.batch_size_large

            if args.learn_type == "random":
                query_idx = np.random.choice(len(X_pool), size=batch_size, replace=False)

            elif args.learn_type == "a_uncertainty":
                query_idx = compute_uncertainty(model, X_pool, batch_size)

            elif args.learn_type == "loss":
                start_batch=time.time()
                query_idx, min_loss, dist_list = compute_final_loss(model, X_train_wa_t0, X_train_wa_t1, X_pool, y_pool,
                                                                    batch_size, j,
                                                                    args.loss_distance, args.loss_factual,
                                                                    outcome_model_c, outcome_model_t, i)

                end_batch=time.time()
                print("Time that took for batch" + str(j) + ":" + str(end_batch-start_batch))
                X_to_add = pd.concat([X_pool, a_pool], axis=1)
                # print(query_idx)
                for q in query_idx:
                    if int(X_to_add.iloc[q]['t']) == 0:
                        X_train_wa_t0 = X_train_wa_t0.append(X_to_add.drop(columns=['t']).iloc[q])
                    else:
                        X_train_wa_t1 = X_train_wa_t1.append(X_to_add.drop(columns=['t']).iloc[q])

            X_train = X_train.append(X_pool.iloc[query_idx])
            y_train = y_train.append(y_pool.iloc[query_idx])
            a_train = a_train.append(a_pool.iloc[query_idx])
            X_train_wa = pd.concat([X_train, a_train], axis=1)

            if model_type == "GP":
                outcome_model_c, outcome_model_t, final_pehe, final_ate = GP(X_train_wa, y_train, X_test_wa, y_test, args.multihead, outcome_model_c, outcome_model_t, seed=i)
            elif model_type == "MLP":
                outcome_model_c, outcome_model_t, final_pehe, final_ate = MLP(X_train_wa, y_train, X_test_wa, y_test, args.multihead, i)
            elif model_type == "DoublyRobust":
                outcome_model_c, outcome_model_t, final_pehe, final_ate = DoublyRobustNN(X_train_wa, y_train, X_test_wa, y_test, args.multihead, i)
            elif model_type == "CF":
                outcome_model_c, outcome_model_t, final_pehe, final_ate = CF(X_train_wa, y_train, X_test_wa, y_test, args.multihead, i)
            elif model_type == "CMGP":
                outcome_model_c, outcome_model_t, final_pehe, final_ate = CausalMGP(X_train_wa, y_train, X_test_wa, y_test, args.multihead, i)
            model = RandomForestClassifier(n_estimators=100)
            model.fit(X_train, a_train['momwhite'])

            X_pool = X_pool.drop(query_idx, axis=0).reset_index(drop=True)
            y_pool = y_pool.drop(query_idx, axis=0).reset_index(drop=True)
            a_pool = a_pool.drop(query_idx, axis=0).reset_index(drop=True)

            if args.save_pehe:
                dist_explore_t = 0
                dist_explore_c = 0
                if j == args.query_nb - 1:
                    dist_explore_t = 0
                    dist_explore_c = 0
                else:
                    dist_explore_t = dist_exploration(X_train, a_train, X_pool, a_pool, 1)
                    dist_explore_c = dist_exploration(X_train, a_train, X_pool, a_pool, 0)
                with open(args.result_path+"errors/"+ "error.csv", "a+") as f:
                    f.write("%s,%s, %s, %s, %d,%d,%d,%.4f,%.4f,%.12f,%.12f\n"%(model_type, args.learn_type, args.loss_distance, args.loss_factual,
                                                    i,j+1,X_train.shape[0],final_pehe,final_ate,dist_explore_t,dist_explore_c))

        X_train.reset_index(drop=True, inplace=True)
        a_train.reset_index(drop=True, inplace=True)
        X_train_wa = pd.concat([X_train, a_train], axis=1)
        X_train_wa.to_csv(args.result_path+'X_train_'+args.learn_type+'_'+str(i)+'.csv', index=False)
        y_train.to_csv(args.result_path+'y_train_'+args.learn_type+'_'+str(i)+'.csv', index=False)

        end = time.time()
        print("For experiment " + str(i) + ", it took: " + str(end-start) + "s.")