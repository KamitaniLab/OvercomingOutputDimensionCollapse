import argparse
import os
from dataclasses import dataclass
from functools import partialmethod

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from overcoming_output_dimension_collapse.sparse_regression.fast_ridge import FastRidge
from overcoming_output_dimension_collapse.sparse_regression.fastl2lir_pro import (
    FastL2LiR,
)
from overcoming_output_dimension_collapse.sparse_regression.solve_theory import (
    SparseRegressionTheory,
)

tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)


# =============================================================================
# Configuration and Utility Functions
# =============================================================================


def create_argument_parser():
    """Create and configure argument parser."""
    parser = argparse.ArgumentParser(description="Run sparse regression experiment")
    parser.add_argument(
        "--d_in", type=int, default=1000, help="Number of input features"
    )
    parser.add_argument(
        "--d_out", type=int, default=1000, help="Number of output targets"
    )
    parser.add_argument(
        "--n_test", type=int, default=1000, help="Number of test samples"
    )
    parser.add_argument(
        "--nonzero_ratio",
        type=float,
        default=0.01,
        help="ratio of nonzero elements in W",
    )
    parser.add_argument(
        "--noise_std", type=float, default=0.1, help="noise standard deviation"
    )
    parser.add_argument("--n_trials", type=int, default=100, help="Number of trials")
    parser.add_argument(
        "--data_type",
        type=str,
        default="baseline",
        choices=["baseline", "gaussian-weight", "correlated-signal", "input-noise"],
        help="Type of data generation",
    )
    parser.add_argument(
        "--rho",
        type=float,
        default=0.5,
        help="Correlation parameter for correlated-signal",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="assets/2_sparse_regression/simulation",
        help="Output directory for results",
    )
    return parser


def setup_experiment_parameters():
    """Setup experiment parameters and training sizes."""
    n_trains = [int(10 * (2 ** (k / 2))) for k in range(20)]
    settings = [
        "select-optimal",
        "without-select-optimal",
    ]
    return n_trains, settings


# =============================================================================
# Data Generation Functions
# =============================================================================


def generate_data(
    data_type, d_in, d_out, n_train, n_test, nonzero_ratio, noise_std, rho=0.5
):
    """Generate synthetic data based on specified type."""
    d_nonzero = int(nonzero_ratio * d_in)

    # Create sparse mask matrix M
    while True:
        M = np.zeros((d_in, d_out))
        for col in M.T:
            col[:d_nonzero] = 1 / np.sqrt(d_nonzero)
            np.random.shuffle(col)
        # Check that the first column has approximately correct norm
        if np.abs(np.sum(M[:, 0] ** 2) - 1) < 1e-10:
            if np.linalg.matrix_rank(M) == min(d_in, d_out):
                break

    if data_type == "baseline":
        # Use M as the weight matrix
        W = M
        X_train = np.random.randn(n_train, d_in)
        X_test = np.random.randn(n_test, d_in)
        Y_train = X_train @ W + np.random.randn(n_train, d_out) * noise_std
        Y_test = X_test @ W + np.random.randn(n_test, d_out) * noise_std

    elif data_type == "gaussian-weight":
        # Use M * random gaussian as weight matrix
        W = M * np.random.randn(d_in, d_out)
        X_train = np.random.randn(n_train, d_in)
        X_test = np.random.randn(n_test, d_in)
        Y_train = X_train @ W + np.random.randn(n_train, d_out) * noise_std
        Y_test = X_test @ W + np.random.randn(n_test, d_out) * noise_std

    elif data_type == "correlated-signal":
        # Use M as weight matrix with correlated input
        W = M
        cov_matrix = np.array(
            [[rho ** abs(i - j) for j in range(d_in)] for i in range(d_in)]
        )
        X_train = np.random.multivariate_normal(
            np.zeros(d_in), cov_matrix, size=n_train
        )
        X_test = np.random.multivariate_normal(np.zeros(d_in), cov_matrix, size=n_test)
        Y_train = X_train @ W + np.random.randn(n_train, d_out) * noise_std
        Y_test = X_test @ W + np.random.randn(n_test, d_out) * noise_std

    elif data_type == "input-noise":
        # Use M as weight matrix, add noise to input
        W = M
        X_train_clean = np.random.randn(n_train, d_in)
        X_test_clean = np.random.randn(n_test, d_in)
        Y_train = X_train_clean @ W
        Y_test = X_test_clean @ W
        X_train = X_train_clean + np.random.randn(n_train, d_in) * noise_std
        X_test = X_test_clean + np.random.randn(n_test, d_in) * noise_std

    else:
        raise ValueError(f"Unknown data type: {data_type}")

    return X_train, X_test, Y_train, Y_test, W


# =============================================================================
# Model Training and Prediction Functions
# =============================================================================


def fit_predict_fastl2lir(X_train, Y_train, X_test, n_feat, alpha):
    """Fit FastL2LiR model and make predictions."""
    model = FastL2LiR()
    model.fit(X_train, Y_train, alpha=alpha, n_feat=n_feat)
    Y_pred = model.predict(X_test)
    return Y_pred, n_feat


def fit_predict_fastridge(X_train, Y_train, X_test, alpha=None):
    """Fit FastRidge model and make predictions."""
    if alpha is None:
        alpha = X_train.shape[1] / 10
    model = FastRidge(alpha=alpha)
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    return Y_pred


# =============================================================================
# Experiment Execution Functions
# =============================================================================


def run_single_trial(
    trial_idx,
    data_type,
    d_in,
    d_out,
    n_train_max,
    n_test,
    nonzero_ratio,
    noise_std,
    rho,
    n_trains,
    settings,
    solver,
):
    """Run a single trial of the experiment."""
    # Generate data for this trial
    X_train, X_test, Y_train, Y_test, W = generate_data(
        data_type, d_in, d_out, n_train_max, n_test, nonzero_ratio, noise_std, rho
    )

    trial_results = {}

    # Fit and predict models for different training sizes
    for n_train in n_trains:
        X_train_sub = X_train[:n_train, :]  # (n_train, d_in)
        Y_train_sub = Y_train[:n_train, :]  # (n_train, d_out)

        # Standardize X and Y
        scalerX = StandardScaler()
        scalerY = StandardScaler()
        X_train_sub_std = scalerX.fit_transform(X_train_sub)  # (n_train, d_in)
        Y_train_sub_std = scalerY.fit_transform(Y_train_sub)  # (n_train, d_out)

        # Transform test data
        X_test_std = scalerX.transform(X_test)  # (n_test, d_in)
        scalerY.transform(Y_test)  # (n_test, d_out)

        # Fit and predict models for different settings
        for setting in settings:
            if setting == "select-optimal":
                p_sel, risk, tau2 = solver.solve_optimal_p_select_and_risk(
                    nonzero_ratio, noise_std, n_train / d_in
                )
                Y_pred_std, n_feat = fit_predict_fastl2lir(
                    X_train_sub_std,
                    Y_train_sub_std,
                    X_test_std,
                    n_feat=int(p_sel * d_in),
                    alpha=d_in / tau2,
                )
            elif setting == "without-select-optimal":
                risk, tau2 = solver.solve_risk_and_tau2_for_p_select(
                    nonzero_ratio, noise_std, n_train / d_in, 1.0
                )
                Y_pred_std = fit_predict_fastridge(
                    X_train_sub_std, Y_train_sub_std, X_test_std, alpha=d_in / tau2
                )
                n_feat = d_in
            else:
                raise ValueError(f"Invalid setting: {setting}")

            # Destandardize predictions
            Y_pred = scalerY.inverse_transform(Y_pred_std)

            # Evaluate performance
            mse = np.mean((Y_pred - Y_test) ** 2)

            # Store results for this trial
            key = (n_train, setting)
            trial_results[key] = {"mse": mse, "n_feat": n_feat, "risk": risk}

    return trial_results


# =============================================================================
# Results Saving Functions
# =============================================================================


def save_results(
    mse_dict,
    n_feat_dict,
    risk_dict,
    d_in,
    d_out,
    n_test,
    nonzero_ratio,
    noise_std,
    n_trials,
    data_type,
    rho,
    output_dir,
):
    """Save experiment results to CSV file."""
    os.makedirs(output_dir, exist_ok=True)

    @dataclass
    class Result:
        d_in: int
        d_out: int
        n_train: int
        n_test: int
        nonzero_ratio: float
        noise_std: float
        setting: str
        n_feat: int
        n_trials: int
        mse: np.ndarray
        estimated_risk: float
        data_type: str
        rho: float

    results = []
    for key in mse_dict.keys():
        n_train, setting = key
        result = Result(
            d_in=d_in,
            d_out=d_out,
            n_train=n_train,
            n_test=n_test,
            nonzero_ratio=nonzero_ratio,
            noise_std=noise_std,
            setting=setting,
            n_feat=n_feat_dict[key],
            n_trials=n_trials,
            mse=np.array(mse_dict[key]),
            estimated_risk=risk_dict[key],
            data_type=data_type,
            rho=rho,
        )
        results.append(result)

    df = pd.DataFrame(results)

    # Save results (append if file exists)
    save_path = f"{output_dir}/results.csv"
    if not os.path.exists(save_path):
        df.to_csv(save_path, index=False)
    else:
        df.to_csv(save_path, mode="a", header=False, index=False)


# =============================================================================
# Main Function
# =============================================================================

if __name__ == "__main__":
    # Parse command line arguments
    parser = create_argument_parser()
    args = parser.parse_args()

    # Extract parameters
    d_in = args.d_in
    d_out = args.d_out
    n_test = args.n_test
    nonzero_ratio = args.nonzero_ratio
    noise_std = args.noise_std
    n_trials = args.n_trials
    data_type = args.data_type
    rho = args.rho
    output_dir = args.output_dir

    # Validate parameters
    assert 0 < nonzero_ratio <= 1, "nonzero_ratio must be between 0 and 1"

    # Setup experiment parameters
    n_trains, settings = setup_experiment_parameters()
    n_train_max = n_trains[-1]

    # Initialize results storage
    mse_dict = {}
    n_feat_dict = {}
    risk_dict = {}

    # Initialize solver
    solver = SparseRegressionTheory()

    # Run experiments
    for trial in tqdm(range(n_trials), disable=False):
        trial_results = run_single_trial(
            trial,
            data_type,
            d_in,
            d_out,
            n_train_max,
            n_test,
            nonzero_ratio,
            noise_std,
            rho,
            n_trains,
            settings,
            solver,
        )

        # Accumulate results
        for key, result in trial_results.items():
            if key not in mse_dict:
                mse_dict[key] = []
                n_feat_dict[key] = result["n_feat"]
                risk_dict[key] = result["risk"]
            mse_dict[key].append(result["mse"])

    # Save results
    save_results(
        mse_dict,
        n_feat_dict,
        risk_dict,
        d_in,
        d_out,
        n_test,
        nonzero_ratio,
        noise_std,
        n_trials,
        data_type,
        rho,
        output_dir,
    )
