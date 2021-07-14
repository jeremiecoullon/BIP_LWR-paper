import numpy as np
import numpy.linalg as LA

class OUProcess:
    """
    Ornstein Uhlenbeck object with mean 0. Choose parameters along with number of random variables
    N, and time step dt.
    Uses a finite difference representation.
    Precision: Precision matrix:
    cholPrecision: Cholesky matrix (upper triangular)
    sample(): sample from process
    """
    def __init__(self, beta, dt, sigma, N):
        self.beta = beta
        self.dt = dt
        self.sigma = sigma
        self.N = N
        self.Precision = self.create_precision(N=N, beta=self.beta, sigma=self.sigma, dt=self.dt)
        # self.cholPrecision = LA.cholesky(self.Precision).T
        self.cov_prior = LA.inv(self.Precision)
        self.cov_chol = LA.cholesky(self.cov_prior)

    def OU_var(self, beta, dt, sigma):
        return (sigma**2 / (2*beta))*(1 - np.exp(-2*beta*dt))

    def create_precision(self, N, beta, sigma, dt):
        """
        OU precision matrix using finite difference
        """
        diag_lower = np.zeros((N, N))
        rng = np.arange(len(diag_lower)-1)
        term_1 = -np.exp(-beta*dt)
        diag_lower[rng+1, rng] = term_1

        # offset upper
        diag_upper = np.zeros((N, N))
        rng = np.arange(len(diag_upper)-1)
        diag_upper[rng, rng+1] = term_1

        # build precision matrix
        term_2 = 1+np.exp(-2*beta*dt)
        scaling_term = (1/(1*self.OU_var(beta=beta, dt=dt, sigma=sigma)))
        Precision = (np.eye(N)*term_2 + diag_lower + diag_upper)
        Precision[0,0] = 1
        Precision[-1, -1] = 1
        return scaling_term*Precision

    def sample(self):
        return self.cov_chol @ np.random.normal(loc=0, scale=1, size=self.N)
        # return LA.solve(self.cholPrecision, np.random.normal(loc=0, scale=1, size=self.N))
