"""A place for modes that aren't in prysm.polynomials"""
import scipy
from prysm.mathops import np

def hadamard_modes_sequence(aperture):
    """Generate a sequence of hadamard modes

    Code adapted from the uasal/lina package
    https://github.com/uasal/lina/blob/main/lina/utils.py

    Notes
    -----
    JNA: It looks to me like this constructs a Hadamard matrix of the order
    equal to the number of actuators in the DM. This means that each vector
    in the Hadamard matrix corresponds to a mode. Hadamard matrices are 
    orthogonal, which means each vector is mutually orthogonal. This should
    mean that the resulting modes (i.e. rows of the hadamard matrix), should
    be orthogonal and span the vector space of the DM.

    Parameters
    ----------
    aperture : ndarray
        binary array denoting the aperture transmission function

    Returns
    -------
    list of ndarrays
        sequence of hadamard modes
    """
    
    # grab DM actuator dimensions
    num_actuators = aperture.sum().astype(int)
    shape_actuators = aperture.shape[0]

    # construct the hadamard matrix of order np2
    np2 = 2**int(np.ceil(np.log2(num_actuators)))
    hmodes = np.array(scipy.linalg.hadamard(np2))
    
    had_modes = []

    inds = np.where(aperture.flatten().astype(int))

    # each vector in the hadamard matrix is a mode
    for hmode in hmodes:
        hmode = hmode[:num_actuators]
        mode = np.zeros((aperture.shape[0]**2))
        mode[inds] = hmode
        had_modes.append(mode)

    had_modes = np.array(had_modes).reshape(np2, shape_actuators, shape_actuators)
    
    return had_modes
