import numpy as np
from scipy.optimize import minimize_scalar, root_scalar
from scipy.special import erf
from typing import Tuple, Union


class SparseRegressionTheory:
    """
    Sparse regression theory solver with clean public API.
    
    This class provides methods to solve sparse regression problems using
    theoretical analysis. The main public methods are:
    - solve_optimal_p_select_and_risk: Solve for optimal selection probability and risk
    - solve_risk_and_tau2_for_p_select: Solve for risk and tau2 for given selection probability
    - solve_pi_given_a_p_select_and_data_scale: Solve for pi given parameters
    - risk_from_p_sel_and_pi: Compute risk from selection probability and pi
    """
    
    def __init__(self, u_max: float = 100.0, u_min: float = 0.0):
        """
        Initialize the sparse regression theory solver.
        
        Args:
            u_max: Maximum value for u parameter in root finding
            u_min: Minimum value for u parameter in root finding
        """
        self._u_max = u_max
        self._u_min = u_min
    
    # =============================================================================
    # Public API Methods
    # =============================================================================
    
    def solve_optimal_p_select_and_risk(
        self, 
        alpha: float, 
        sigma: float, 
        data_scale: float
    ) -> Tuple[float, float, float]:
        """
        Solve for the optimal selection probability and corresponding minimal risk.
        
        This method optimizes the selection probability p_select to minimize
        the prediction risk, given the sparsity parameter, noise level, and data scale.
        
        Args:
            alpha: Sparsity parameter (0 < alpha <= 1)
            sigma: Noise standard deviation (sigma > 0)
            data_scale: Data scale parameter (data_scale > 0)
        
        Returns:
            Tuple containing:
                optimal_p_select: The optimal selection probability
                minimal_risk: The corresponding minimal risk value
                optimal_tau2: The corresponding tau2 value
                
        Raises:
            ValueError: If parameters are outside valid ranges or optimization fails
        """
        # Input validation
        if not (0 < alpha <= 1):
            raise ValueError(f"alpha must be between 0 and 1, got {alpha}")
        if not (sigma > 0):
            raise ValueError(f"sigma must be positive, got {sigma}")
        if not (data_scale > 0):
            raise ValueError(f"data_scale must be positive, got {data_scale}")
        
        def objective_function(p_select: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
            """
            Objective function for risk minimization.
            
            Args:
                p_select: Selection probability value(s)
                
            Returns:
                Risk value(s) corresponding to the given p_select
            """
            if isinstance(p_select, float):
                return self.solve_risk_and_tau2_for_p_select(alpha, sigma, data_scale, p_select)[0]
            elif isinstance(p_select, np.ndarray):
                return np.array([self.solve_risk_and_tau2_for_p_select(alpha, sigma, data_scale, p)[0] 
                               for p in p_select])
            else:
                raise ValueError(f"Invalid p_select type: {type(p_select)}")
        
        # Set bounds for optimization (alpha <= p_select <= 1)
        bounds = (alpha, 1.0)

        try:
            result = minimize_scalar(objective_function, bounds=bounds, method='bounded')
        except Exception as e:
            raise ValueError(f"Optimization failed: {e}")

        # Compare with naive solution (p_select = 1)
        naive_risk, naive_tau2 = self.solve_risk_and_tau2_for_p_select(alpha, sigma, data_scale, 1.0)
        
        if result.success:
            if result.fun < naive_risk:
                optimal_p_select = result.x
                minimal_risk, optimal_tau2 = self.solve_risk_and_tau2_for_p_select(
                    alpha, sigma, data_scale, optimal_p_select)
                return optimal_p_select, minimal_risk, optimal_tau2
            else:
                return 1.0, naive_risk, naive_tau2
        else:
            raise ValueError(f"Optimization failed at data_scale={data_scale}, "
                           f"alpha={alpha}, sigma={sigma}. "
                           f"Optimization message: {result.message}")
    
    def solve_risk_and_tau2_for_p_select(
        self, 
        alpha: float, 
        sigma: float, 
        data_scale: float,
        p_select: float
    ) -> Tuple[float, float]:
        """
        Solve for the risk and tau2 for a given selection probability.
        
        This method calculates the prediction risk and tau2 parameter when a 
        specific selection probability p_select is used, given the sparsity 
        parameter, noise level, and data scale.
        
        Args:
            alpha: Sparsity parameter (0 < alpha <= 1)
            sigma: Noise standard deviation (sigma > 0)
            data_scale: Data scale parameter (data_scale > 0)
            p_select: Selection probability (0 < p_select <= 1)
            
        Returns:
            Tuple containing:
                risk: The computed risk value
                tau2: The computed tau2 parameter
                
        Raises:
            ValueError: If parameters are outside valid ranges
        """
        # Input validation
        if not (0 < alpha <= 1):
            raise ValueError(f"alpha must be between 0 and 1, got {alpha}")
        if not (0 < p_select <= 1):
            raise ValueError(f"p_select must be between 0 and 1, got {p_select}")
        if not (sigma > 0):
            raise ValueError(f"sigma must be positive, got {sigma}")
        if not (data_scale > 0):
            raise ValueError(f"data_scale must be positive, got {data_scale}")

        # Solve for pi using the given parameters
        pi_sol = self.solve_pi_given_a_p_select_and_data_scale(alpha, p_select, data_scale, sigma)

        # Compute risk parameters
        sigma2 = 1 - pi_sol + sigma**2
        tau2 = pi_sol / sigma2
        gamma = p_select / data_scale
        
        # Calculate the prediction error
        risk = self._prediction_error_numeric(sigma2, tau2, gamma)
        return risk, tau2
    
    def solve_pi_given_a_p_select_and_data_scale(
        self, 
        a: float, 
        p_select: float, 
        data_scale: float, 
        sigma: float
    ) -> float:
        """
        Solve for pi given the sparsity parameter, selection probability, data scale, and noise level.
        
        This method computes the probability pi that a feature is selected under the given
        sparsity and selection constraints.
        
        Args:
            a: Proportion of non-zero weights (0 < a <= 1)
            p_select: Selection probability (0 < p_select <= 1)
            data_scale: Data scale parameter (data_scale > 0)
            sigma: Noise standard deviation (sigma > 0)
            
        Returns:
            pi: The computed selection probability
            
        Raises:
            ValueError: If parameters are outside valid ranges or optimization fails
        """
        # Input validation
        if not (0 < a <= 1):
            raise ValueError(f"a must be between 0 and 1, got {a}")
        if not (0 < p_select <= 1):
            raise ValueError(f"p_select must be between 0 and 1, got {p_select}")
        if not (sigma > 0):
            raise ValueError(f"sigma must be positive, got {sigma}")
        if not (data_scale > 0):
            raise ValueError(f"data_scale must be positive, got {data_scale}")

        if p_select == 1:
            return 1.0
        
        def f_u(u: float) -> float:
            """
            Objective function for finding the optimal threshold u.
            
            Args:
                u: Threshold parameter (u = t / sqrt(n))
                
            Returns:
                Function value to be minimized
            """
            sqrt_term = np.sqrt(data_scale / a)
            sigma_term = np.sqrt(1 + sigma ** 2)
            
            pi = (self._cdf(-(u - sqrt_term) / sigma_term) + 
                  self._cdf(-(u + sqrt_term) / sigma_term))
            pi_0 = 2 * self._cdf(-u / sigma_term)
            
            return -p_select + a * pi + (1 - a) * pi_0
        
        # Solve for u using root finding
        u_min = self._u_min
        u_max = self._u_max
        f_u_min = f_u(u_min)
        f_u_max = f_u(u_max)
        
        try:
            if f_u_min * f_u_max < 0:
                # Use bracketing method if signs are different
                u_sol = root_scalar(f_u, bracket=[u_min, u_max], method='brentq').root
            else:
                # Use Newton's method if signs are the same
                result = root_scalar(f_u, x0=u_max, method='newton')
                if result.converged:
                    u_sol = result.root
                else:
                    raise ValueError(f"Newton's method failed to converge for u_sol")
        except Exception as e:
            raise ValueError(f"Root finding failed: {e}")
        
        # Compute final pi value
        sqrt_term = np.sqrt(data_scale / a)
        sigma_term = np.sqrt(1 + sigma ** 2)
        pi_sol = (self._cdf(-(u_sol - sqrt_term) / sigma_term) + 
                  self._cdf(-(u_sol + sqrt_term) / sigma_term))
        
        return pi_sol
    
    def risk_from_p_sel_and_pi(self, p_sel: float, pi: float, sigma: float, data_scale: float) -> float:
        """
        Compute the risk from the selection probability and the probability of selection.
        
        Args:
            p_sel: Selection probability
            pi: Probability of selection
            sigma: Noise standard deviation
            data_scale: Data scale parameter
            
        Returns:
            The computed risk value
        """
        sigma_eff2 = 1 - pi + sigma**2
        tau2 = pi / sigma_eff2
        gamma = p_sel / data_scale
        return self._prediction_error_numeric(sigma_eff2, tau2, gamma)
    
    # =============================================================================
    # Private Helper Methods
    # =============================================================================
    
    def _cdf(self, t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Cumulative distribution function of the standard normal distribution.
        
        Args:
            t: Input value(s) for which to compute the CDF
            
        Returns:
            CDF value(s) of the standard normal distribution
        """
        return 0.5 * (1 + erf(t / np.sqrt(2)))

    def _ridge_risk_numeric(self, tau2: float, gamma: float) -> float:
        """
        Compute the ridge risk numerically.
        
        This method calculates the ridge risk using the analytical formula
        derived from the ridge regression theory.
        
        Args:
            tau2: Signal-to-noise ratio parameter
            gamma: Regularization parameter
            
        Returns:
            The computed ridge risk value
        """
        discriminant = (tau2 * (gamma - 1) - gamma)**2 + 4 * gamma**2 * tau2
        numerator = tau2 * (gamma - 1) - gamma + np.sqrt(discriminant)
        denominator = 2 * gamma
        return numerator / denominator
    
    def _prediction_error_numeric(self, sigma2: float, tau2: float, gamma: float) -> float:
        """
        Compute the prediction error numerically.
        
        This method calculates the prediction error based on the ridge risk
        and noise variance.
        
        Args:
            sigma2: Noise variance
            tau2: Signal-to-noise ratio parameter
            gamma: Regularization parameter
            
        Returns:
            The computed prediction error
        """
        return sigma2 * (self._ridge_risk_numeric(tau2, gamma) + 1)