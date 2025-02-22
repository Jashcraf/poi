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


def _fourier_modes_sequence(Nacts, Nmodes=None, which="cos"):

    max_frequency = Nacts // 2 # cy / pupil
    u = np.linspace(-1/2, 1/2, Nacts)
    uu, vv = np.meshgrid(u, u)

    if Nmodes is None:
        Nmodes = Nacts // 2

    modes = []

    frequency = np.linspace(0, max_frequency, Nmodes)
    for f in frequency:

        if which == "cos":
            mode = np.cos(2 * np.pi * f * uu)

        elif which == "sin":
            mode = np.sin(2 * np.pi * f * uu)

        modes.append(mode)

    return modes


def create_annular_focal_plane_mask(npsf, psf_pixelscale, 
                                    irad, orad,  
                                    edge=None,
                                    shift=(0,0), 
                                    rotation=0):
    x = (np.linspace(-npsf/2, npsf/2-1, npsf) + 1/2)*psf_pixelscale
    x,y = np.meshgrid(x,x)
    r = np.hypot(x, y)
    mask = (r > irad) * (r < orad)

    if edge is not None:
        mask *= (x > edge)
    
    mask = scipy.ndimage.rotate(mask, rotation, reshape=False, order=0)

    return mask


def fourier_modes_sequence(Nact, Nimg, psf_pixelscale_lamD, iwa, owa, 
                         edge=None,
                         rotation=0, 
                         fourier_sampling=0.75, 
                         which='both', 
                         return_fs=False):
    """Create a sequence of fourier modes inside a region defined by an inner
    working angle (iwa) and outer working angle (owa)

    Parameters
    ----------
    Nact : int
        number of actuators across a DM
    Nimg : int
        number of samples across an image
    psf_pixelscale_lamD : float
        pixelscale in lambda / D
    iwa : float
        inner working angle in lambda / D
    owa : float
        outer working angle in lambda / D
    edge : float, optional
        A knife-edge limit on the control region, starts from left to right, by default None
    rotation : int, optional
        degrees by which to rotate the control region, by default 0
    fourier_sampling : float, optional
        TODO: Verify that this is correct
        how many sinusoids to sample each pixel by, by default 0.75
    which : str, optional
        whether to return sines, cosines, or a mixture of both, by default 'both'
    return_fs : bool, optional
        returns the computed frequencies if true, by default False

    Returns
    -------
    ndarray
        array of fourier modes
    """
    
    nfg = int(np.round(Nimg * psf_pixelscale_lamD/fourier_sampling))

    # ensure even
    if nfg%2==1:
        nfg += 1

    yf, xf = (np.indices((nfg, nfg)) - nfg//2 + 1/2) * fourier_sampling
    fourier_cm = create_annular_focal_plane_mask(nfg, fourier_sampling, iwa-fourier_sampling, owa+fourier_sampling, 
                                                 edge=edge, rotation=rotation)
    
    ypp, xpp = (np.indices((Nact, Nact)) - Nact//2 + 1/2)
    sampled_fs = np.array([xf[fourier_cm], yf[fourier_cm]]).T
    fourier_modes = []

    for i in range(len(sampled_fs)):

        fx = sampled_fs[i,0]
        fy = sampled_fs[i,1]

        if which=='both' or which=='cos':
            fourier_modes.append(np.cos(2 * np.pi * (fx*xpp + fy*ypp) / Nact))

        if which=='both' or which=='sin':
            fourier_modes.append(np.sin(2 * np.pi * (fx*xpp + fy*ypp) / Nact))
    
    if return_fs:
        return np.array(fourier_modes), sampled_fs
    else:
        return np.array(fourier_modes)


def poke_mode_sequence(mask, amplitude=1.):

    modes = []
    zero_mask = np.zeros_like(mask)

    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):

            # poke
            poke_mode = np.copy(zero_mask)
            poke_mode[i, j] = amplitude
            modes.append(poke_mode)

    return modes