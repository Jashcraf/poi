from prysm.mathops import np, fft
from prysm.propagation import focus_fixed_sampling, focus_fixed_sampling_backprop
from prysm import coordinates, geometry


def log_sum_exp(x, alpha=1):
    """
    LogSumExp function, a smooth approximation to max()

    Parameters
    ----------
    x: ndarray
        data to find maximum value of
    alpha: float, optional
        "steepness" parameter for the function. Larger values
        tend to have higher accuracy for small x, but can
        result in underflow errors
    """
    return 1/alpha * np.log(np.sum(np.exp(alpha * x)))


def softmax(x, alpha=1):
    """
    Softmax function, returns a probability distribution of the "likelihood"
    of every value to be the maximum value. Happens to be gradient of
    log_sum_exp
    
    Parameters
    ----------
    x: ndarray
        data to find maximum value of
    alpha: float, optional
        "steepness" parameter for the function. Larger values
        tend to have higher accuracy for small x, but can
        result in underflow errors
    """
    return np.exp(alpha * x) / np.sum(np.exp(alpha * x))


class LogSumExp:
    def __init__(self, target=0, alpha=1):
        """Object interface for the LogSumExp cost function, with
        'forward' and 'reverse' method for use in models with 
        analytic gradients.
        """
        self.target = target
        self.alpha = alpha

    def forward(self, x):
        return log_sum_exp(self.target - x, alpha=self.alpha)

    def reverse(self, x):
        return softmax(self.target - x, alpha=self.alpha) 


class MaxContrast:
    def __init__(self, target=0, alpha=1):
        """
        Targets maximum value, uses softmax to approximate gradient
        """
        self.target = target
        self.alpha = alpha

    def forward(self, x):
        return np.max(self.target - x)

    def reverse(self, x):
        return softmax(self.target - x, alpha=self.alpha)


class MeanSquaredErrorLinear:
    def __init__(self, target=0, alpha=1.):
        """Mean squared error cost function with a linear
        penalty. Sign changes when constraint target is satisfied.
        In english, this means you are asking the optimizer:
        "Hey please get to 'target', but if you can do better, that's great"

        Parameters
        ----------
        target: float
            Target contrast. These are implicitly converted to squared unites.
            If you give it 1e-2, it will target 1e-4 so that 1e-2 is achieved.
        alpha: float
            Weight to multiply constraint by. Can be negative
            to flip the sign convention
        """
        self.target = target ** 2
        self.alpha = alpha

    def forward(self, x):
        err = x
        mse_mag = np.mean(err**2)
        return (mse_mag - self.target) * self.alpha

    def reverse(self, x):
        err = x 
        mse_grad = 2 * err / x.size 
        return mse_grad * self.alpha 


class MeanSquaredErrorQuadratic:
    def __init__(self, target=0, alpha=1.):
        """Mean squared error cost function with a linear
        penalty. Sign changes when constraint target is satisfied
        In english, this means you are asking the optimizer:
        "Hey please get to 'target', but if you can do better, don't"

        Parameters
        ----------
        target: float
            Target contrast, not MSE units.
        alpha: float
            Weight to multiply constraint by. Can be negative
            to flip the sign convention
        """
        self.target = target
        self.alpha = alpha

    def forward(self, x):
        err = x - self.target
        mse_mag = np.sum(err**2)
        return mse_mag * self.alpha

    def reverse(self, x):
        err = x - self.target 
        mse_grad = 2 * err#  / x.size 
        return mse_grad * self.alpha 

class PNorm:
    def __init__(self, target=0, alpha=10):
        self.target = target
        self.alpha = alpha

    def forward(self, x):
        exponent = 1 / self.alpha
        self.p_norm = (np.sum(x) ** self.alpha) ** exponent
        return self.target - self.p_norm

    def reverse(self, x):
        return x ** (self.alpha - 1) / (self.p_norm ** (self.alpha-1)) 

class CoreThroughput:
    def __init__(self, target=0):
        """ Core Throughput Maximization, negative sign applied to both
        forward and reverse to maximize core throughput instead of 
        minimize

        Parameters
        ----------
        target: ndarray
            Core window on-axis to evaluate the core throughput
            of a coronagraph. 

        """
        self.target = target

    def forward(self, x):
        """
        Use LSE to approximate the maximum in the core window

        Parameters
        ----------
        x: ndarray
            Image plane intensity
        """

        return -1 * np.sum(x - self.target)

    def reverse(self, x):
        """
        Use softmax to backpropagate the smooth maximum in the core window

        Parameters
        ----------
        x: ndarray
            Image plane intensity
        """
        return -1 * (x - self.target)

