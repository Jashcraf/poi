import numpy as np
import matplotlib.pyplot as plt

def threshold(array, threshold, threshold_value=0):
    """


    Parameters
    ----------
    array : numpy.ndarray
        image to threshold
    threshold : float
        value to threshold below
    threshold_value : float
        value to set thresholded values to
        
    Returns
    -------
    numpy.ndarray
        thresholded array
    """

    array_thresholded = np.copy(array)
    array_thresholded[array<threshold] = threshold_value
    return array_thresholded


def mean_squared_error(x1, x2, norm=1):
    """return the mean squared error between two signals

    Parameters
    ----------
    x1 : ndarray
        signal 1, the "observation"
    x2 : ndarray
        signal 2, the "prediction"
    norm : int, optional
        optional normalization, by default 1

    Returns
    -------
    float
        mean squared error of the two signals
    """
    return np.mean(((x1 - x2)**2) / norm)



def phase_unwrap_2d(phase_wrapped):

    """phase unwrapping routine based on the phaseunwrap2d.go script in IDL and the following proceedings:
    M.D. Pritt; J.S. Shipman, "Least-squares two-dimensional phase unwrapping using FFT's",
    IEEE Transactions on Geoscience and Remote Sensing ( Volume: 32, Issue: 3, May 1994),
    DOI: 10.1109/36.297989

    Uses a finite differences approach to determine the partial derivative of the wrapped phase in x and y,
    then solves the solution in the fourier domain

    TODO: Test this function against the prior in IDL, it doesn't appear to reconstruct phase well

    Parameters
    ----------
    phase_wrapped : numpy.ndarray
        array containing 2D signal to unwrap

    Returns
    -------
    numpy.ndarray
        unwrapped phase
    """

    imsize = phase_wrapped.shape
    M = imsize[0]
    N = imsize[1]

    Nmirror = 2 * (N )
    Mmirror = 2 * (M )

    phmirror = np.ones([Mmirror,Nmirror])

    # Quadrant 3
    phmirror[:M,:N] = phase_wrapped

    # First mirror reflection Quadrant 2
    phmirror[M:,:N] = np.flipud(phase_wrapped)

    # Second mirror reflection Quadrant 4
    phmirror[:M,N:] = np.fliplr(phase_wrapped)

    # Final reflection Quadrant 1
    phmirror[M:,N:] = np.flipud(np.fliplr(phase_wrapped))

    phroll = np.zeros_like(phmirror)
    phroll[:M,:N-1] = phmirror[:M,1:N]
    phroll[:M,N-1] = phmirror[:M,0]
    deltafd = phroll-phmirror

    pluspi = np.pi*np.ones_like(phmirror)
    mask = (deltafd > pluspi).astype(int)
    deltafd = deltafd - mask*2*np.pi
    negpi = -pluspi
    mask = (deltafd < negpi).astype(int)
    deltafd = deltafd + mask * 2 * np.pi
    deltafdx = deltafd

    # compute forward difference
    phroll = np.zeros_like(phmirror)
    phroll[:M-1,:N] = phmirror[1:M,:N]
    phroll[M,:N] = phmirror[0,:N]
    deltafd = phroll - phmirror

    pluspi = np.pi*np.ones_like(phmirror)
    mask = (deltafd > pluspi).astype(int)
    deltafd = deltafd - mask*2*np.pi
    negpi = -pluspi
    mask = (deltafd < negpi).astype(int)
    deltafd = deltafd + mask * 2 * np.pi
    deltafdy = deltafd

    # Solve system of equations formed by min LS -> phi
    D_n = np.fft.fft2(deltafdx)
    D_m = np.fft.fft2(deltafdy)
    inc_n = 2 * np.pi / Nmirror
    inc_m = 2 * np.pi / Mmirror

    nn = np.ones([Mmirror,1]) @ (np.arange(Nmirror))[np.newaxis]
    mm = np.ones([Nmirror,1]) @ (np.arange(Mmirror))[np.newaxis]
    mm = mm.transpose()
    print(mm.shape)
    i = 1j
    mult_n = np.ones([Mmirror,Nmirror]) - np.exp(-nn * i * inc_n)
    mult_m = np.ones([Mmirror,Nmirror]) - np.exp(-mm * i * inc_m)
    divisor = (np.cos(mm*inc_m) + np.cos(nn*inc_n) - np.ones([Mmirror,Nmirror])*2)*2
    divisor[0,0] = 1
    phi = (D_n*mult_n + D_m*mult_m) / divisor
    phi[0,0] = 0
    phi = np.fft.ifft2(phi)[:M,:N]
    phout = np.real(phi)
    return phout


def azimuthal_average(image, center=None, angle_range=[-45, 45]):
    """
    Calculate the azimuthal average of an image over a specified angular range.
    Thanks to Claude AI by Anthropic for the function
    
    Parameters:
    -----------
    image : 2D numpy array
        The input image
    center : tuple, optional
        The (x,y) coordinates of the center. If None, uses the image center
    angle_range : tuple, optional
        The (min_angle, max_angle) in degrees for averaging. If None, uses full 360°
        
    Returns:
    --------
    radial_profile : numpy array
        The azimuthally averaged radial profile
    r_bins : numpy array
        The radial distances corresponding to the profile points
    """
    
    # Get image dimensions and center
    y, x = np.indices(image.shape)
    if center is None:
        center = ((x.max() - x.min()) / 2., (y.max() - y.min()) / 2.)
    
    # Calculate radius and angle for each pixel
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    theta = np.degrees(np.arctan2(y - center[1], x - center[0]))
    
    # Make theta range from 0 to 360
    theta = np.where(theta < 0, theta + 360, theta)
    
    # Set up angle range
    if angle_range is None:
        min_angle, max_angle = 0, 360
    else:
        min_angle, max_angle = angle_range
        # Ensure angles are in 0-360 range
        min_angle = min_angle % 360
        max_angle = max_angle % 360
    
    # Create mask for angular range
    if min_angle <= max_angle:
        mask = (theta >= min_angle) & (theta <= max_angle)
    else:  # Handle case where range crosses 0°
        mask = (theta >= min_angle) | (theta <= max_angle)
    
    # Apply mask to image and radius array
    masked_image = image[mask]
    masked_r = r[mask]
    
    # Define radius bins
    r_max = np.max(r)
    r_bins = np.linspace(0, r_max, min(int(r_max), 100))
    
    # Calculate radial profile
    radial_profile = np.zeros_like(r_bins)
    for i in range(len(r_bins)-1):
        r_mask = (masked_r >= r_bins[i]) & (masked_r < r_bins[i+1])
        if r_mask.any():
            radial_profile[i] = masked_image[r_mask].mean()
    
    return radial_profile, r_bins
