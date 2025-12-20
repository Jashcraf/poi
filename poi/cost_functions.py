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
        return log_sum_exp(x - self.target, alpha=self.alpha)

    def reverse(self, x):
        return softmax(x - self.target, alpha=self.alpha) 


class MeanSquaredError:
    def __init__(self, target=0):
        self.target = target

    def forward(self, x):
        return np.mean((x - self.target) ** 2)

    def reverse(self, x):
        return 2 * (x - self.target) / x.size


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

        return -1 * log_sum_exp(x - self.target)

    def reverse(self, x):
        """
        Use softmax to backpropagate the smooth maximum in the core window

        Parameters
        ----------
        x: ndarray
            Image plane intensity
        """
        return -1 * softmax(x - self.target)

