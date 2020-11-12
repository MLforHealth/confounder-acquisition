import pandas as pd
import numpy as np
import random
import argparse
import itertools
import os
from sklearn.model_selection import train_test_split

# Set parameters
parser = argparse.ArgumentParser()

parser.add_argument('--data_x_path', '-x', type=str,
                    default='../data/IHDP/ihdp_x.csv')
parser.add_argument('--data_output_path', '-d', type=str,
                    default='../data/datasets/')
parser.add_argument('--exp_nb', type=int, default=10)
parser.add_argument('--feat_spec',
                    type=str, nargs='+')
parser.add_argument('--confounders',
                    type=str, nargs='+')
parser.add_argument('--missing_confounders',
                    type=str, nargs='+')
parser.add_argument('--beta_spec',
                    type=float, nargs='+')
parser.add_argument('--epsilon',
                    type=float, nargs='+')
parser.add_argument('--sigma',type=float, default=1)
parser.add_argument('--missing',
                    type=float, nargs='+')
parser.add_argument('--alpha0', nargs='+', type=float)
parser.add_argument('--alpha1', nargs='+', type=float)
parser.add_argument('--prop', type=float, default=1)
parser.add_argument('--format', '-f', type=str,
                    default='npy')
parser.add_argument('--a_extreme', type=str, default='original')
parser.add_argument('--corr', type=float, default=0)
parser.add_argument('--random_frac', type=float, default=1)

args = parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

def generatebeta():
    """Function to generate beta.
    Returns:
        np.array: generated beta.
    """
    # Initialize beta
    beta = np.zeros(26)
    # Set different values
    values = [0, 0.1, 0.2, 0.3, 0.4]

    # Fill in beta
    for i in range(0, 6):
        beta[i] = values[np.where(np.random.multinomial(1,
                                                        [0.5, 0.125,
                                                         0.125, 0.125,
                                                         0.125]) == 1)[0][0]]
    for i in range(6, 25):
        beta[i] = values[np.where(np.random.multinomial(1,
                                                        [0.6, 0.1, 0.1, 0.1,
                                                         0.1]) == 1)[0][0]]
    return beta


def clip(x, m, M):
    """Clip function to avoid creating probability values being greater or
    smaller than certain values.

    Args:
        x (float): value to clip.
        m (float): min value accepted.
        M (float): max value accepted.

    Returns:
        np.array: generated beta.
    """
    return min(max(x, m), M)

def generate_a_original(df, col, random_frac, seed=0):
    """Function to replace a sample of A values with random values.

    Args:
        df (pd.Dataframe): data frame of covariates of all samples.
        col (string): column name of A.
        random_frac (float): fraction of A values that are randomly generated.
        seed (int): random seed.
    Returns:
        np.array: generated A.
    """
    seed_everything(seed)
    a_prop = df[col].mean()
    df['a_mask'] = np.random.binomial(1, random_frac, df.shape[0])
    def simulate_masked_a(x):
        if x['a_mask'] == 0:
            return x[col]
        else:
            return np.random.binomial(1, a_prop, 1)[0]
    a = df.apply(simulate_masked_a, axis=1)
    return a

def generate_a_function(df, col, col_other, corr, seed=0):
    """Function to generate A such that it has certain correlation with another
        covariate in X.

    Args:
        df (pd.Dataframe): data frame of covariates of all samples.
        col (string): column name of A.
        col_other (string): column name of the other covariate.
        corr (float): correlation coefficient.
        seed (int): random seed.
    Returns:
        np.array: generated A.
    """
    seed_everything(seed)
    a_count = df[col].sum()
    df_other = df[col_other]
    mean_other = np.mean(df_other, axis=0)
    cov_other = np.cov(df_other, rowvar=0)
    sigma_x = np.sqrt(cov_other)
    conditional_mu = corr * 1 / sigma_x * (df_other.values - mean_other)
    conditional_cov = 1 - corr**2
    simulated_a = np.array([np.random.normal(c_mu, conditional_cov, 1)[0] for c_mu in conditional_mu])
    a_cutoff = simulated_a[np.argsort(simulated_a)[-a_count]]
    a = np.where(simulated_a >= a_cutoff, 1, 0)
    return a


def generatetreatments(X, confounders, epsilon):
    """Function to generate treatments.

    Args:
        X ((pd.Series): All available features.
        confounders (list(str)): list of features that are confounders.
        epsilon (list(float)): list of coefficients for confounders.
    Returns:
        int: Treatment assignment (either 0 or 1).
    """
    # Initialize confounders and coefficients
    X = X[confounders]

    # Generate probabilities
    p = clip(np.matmul(X, np.transpose(epsilon)), 0.005, 0.995)

    # Determine treatments
    t = np.random.binomial(1, p, 1)[0]

    return t


def generateoutcomes(X, beta, feat_spec, beta_spec, sigma=1, w=0.5, o=0, **args):
    """Function to generate outcomes.

    Args:
        X (pd.Series): All available features.
        beta (np.array): Non-adjusted impact of features on the outcome.
        feat_spec (list(str)): Names of confounders with specified impact.
        beta_spec (list(float)): Specified impact of selected features.
        sigma (float): Affects the noise in the generated data.
        w (float): Affects the outcome values when the treatment is absent.
        o (float): Affects the outcome values under treatment.

    Returns:
        tuple of 4 floats: Outcomes for t=0, t=1 without noise
                           (without the impact of sigma), then the outcomes with noise.
    """
    # Initialize variables
    sigma = sigma
    w = w
    o = o
    spec_idx = [X.index.get_loc(feat) for feat in feat_spec if feat in X]
    X_spec = X[feat_spec]
    X_non_spec = X[set(range(len(X))) - set(spec_idx)]
    beta = np.array(beta)[list(set(range(len(X))) - set(spec_idx))]
    beat_spec = np.array(beta_spec)

    # Generate outcomes
    
    t0 = np.exp(np.matmul(X_non_spec + w, np.transpose(beta))+np.matmul(X_spec, np.transpose(beta_spec)))
    t1 = (np.matmul(X_non_spec + w, np.transpose(beta))+np.matmul(X_spec, np.transpose(beta_spec))) - o
    y_t0 = np.random.normal(t0, sigma, 1)[0]
    y_t1 = np.random.normal(t1, sigma, 1)[0]

    return (t0, t1, y_t0, y_t1)

def generatemissingidx(X, missing_confounders, missing):
    """Function to generate missing row index.

    Args:
        X (pd.Series): All available features.
        missing_confounders (list(str)): names of missing confounders.
        missing (float): Overall missing rate.

    Returns:
        list(int): row indices of X with missing confounders.
    """
    # Compute missingness probabilities
    n_missing = len(missing_confounders)
    X_missing_val = X[missing_confounders].sum(axis=1)
    X_missing_val_count = X_missing_val.value_counts(normalize=True)
    base = 0
    for (missing_idx, missing_val) in zip(X_missing_val_count.index, X_missing_val_count.values):
        base += (n_missing - missing_idx + 1) * missing_val
    base_missing = missing/base

    # Generate probabilities
    X_missing_prob = X_missing_val.apply(lambda x: (n_missing - x + 1)*0.2 + np.random.normal()*0.5)

    # Determine missingness
    num_missing = round(X.shape[0]*(1-missing))
    X_missingness = np.argpartition(X_missing_prob, num_missing)[num_missing:]

    return X_missingness


def factual_y(row):
    """Function to select the factual outcome.

    Args:
        row (pd.Series): Used to select the factual outcome.

    Returns:
        float: Factual outcome.
    """
    if row['t'] == 0:
        return row['y_t0']
    else:
        return row['y_t1']


def counterfactual_y(row):
    """Function to select the counterfactual outcome.

    Args:
        row (pd.Series): Used to select the counterfactual outcome.

    Returns:
        float: Counterfactual outcome.
    """
    if row['t'] == 0:
        return row['y_t1']
    else:
        return row['y_t0']

def create_all(X, t, y, missing_confounders, missing_idx):
    """Function to create the datasets.

    Args:
        X (pd.DataFrame): Feature set.
        t (pd.DataFrame): Treatment set.
        y (pd.DataFrame): Outcome set.
        missing_confounders (list(str)): names of missing confounders.
        missing_idx (np.array(int)): index of rows with missing confounders.

    Returns:
        tuple of 3 pd.DataFrames: Generated datasets in the following order: features without
                                  mask, features with mask, outcomes.
    """
    # Create t
    t_cp = t.copy()

    # Create X and X_mask
    X_cp = X.copy()
    X_cp['t'] = t_cp
    X_mask = X_cp.copy()
    missing_col_idx = [X_mask.columns.get_loc(feat) for feat in missing_confounders]
    X_mask.iloc[missing_idx,missing_col_idx] = np.nan

    # Create y
    y_cp = y.copy()
    y_cp['t'] = t_cp
    y_cp['y_factual'] = y_cp.apply(factual_y, axis=1)
    y_cp['y_counterfactual'] = y_cp.apply(counterfactual_y, axis=1)

    y_cp = y_cp[['y_factual', 'y_counterfactual',
               'mu0', 'mu1', 't']]

    return (X_cp, X_mask, y_cp)


# MAIN
if __name__ == '__main__':
    seed_everything(0)

    # Load raw datasets and set parameter lists
    data_x = pd.read_csv(args.data_x_path)

    beta_spec_list = args.beta_spec
    feat_spec_list = args.feat_spec
    confounders_list = args.confounders
    missing_list = args.missing
    missing_confounders_list = args.missing_confounders
    epsilon_list = args.epsilon

    # Generate A values
    if args.a_extreme == 'original':
        data_x['momwhite'] = generate_a_original(data_x.copy(), 'momwhite', args.random_frac)
    else:
        data_x['momwhite'] = generate_a_function(data_x.copy(), 'momwhite', 'bw', args.corr)

    # The BIG FOR LOOP
    for i in range(args.exp_nb):
        np.random.seed(i)
        beta = generatebeta()

        # Generate y
        data_y = pd.DataFrame(columns=['mu0', 'mu1',
                                        'y_t0', 'y_t1'])
        np.random.seed(i)
        (data_y['mu0'],
        data_y['mu1'],
        data_y['y_t0'],
        data_y['y_t1']) = zip(*data_x.apply(lambda row: generateoutcomes(row,
                                                                        beta=beta,
                                                                        feat_spec = feat_spec_list,
                                                                        beta_spec = beta_spec_list,
                                                                        sigma=args.sigma),
                                            axis=1))

        # Generate t
        data_t = pd.DataFrame(columns=['t'])
        data_t['t'] = data_x.apply(lambda row: generatetreatments(row,
                                                                    confounders=confounders_list,
                                                                    epsilon=epsilon_list),
                                    axis=1)


        # Generate Missingness
        for missing in missing_list:
            # Generate masks
            missing_idx = generatemissingidx(data_x, missing_confounders_list, missing)
            # Generate datasets
            X_all, X_mask_all, y_all = create_all(data_x, data_t, data_y, missing_confounders_list, missing_idx)
            data_name = ['X_all','X_mask_all','y_all']
            data_to_save = [X_all, X_mask_all, y_all]

            # Save datasets
            out_dir = os.path.join(args.data_output_path,'missing_' + str(missing))
            try:
                os.makedirs(out_dir)
            except OSError:
                if not os.path.isdir(out_dir):
                    raise
            for dn, ds in zip(data_name, data_to_save):
                if args.format == 'csv':
                    ds.to_csv(os.path.join(out_dir, dn+'_'+str(i)+'.csv'), index=False)
                elif args.format == 'npy':
                    np.save(os.path.join(out_dir,dn+'_'+str(i)+'.npy'), ds)

            print('Saved datasets for seed', i)
