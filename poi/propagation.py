from prysm.mathops import np, fft

def _angular_spectrum_transfer_function(shape, wvl, dx, z):
    """init the transfer function of free space

    Parameters
    ----------
    shape : float
        shape of the array to propagate, array.shape[0] assuming square arrays
    wvl : float
        wavelength in units of microns
    dx : float
        sample size (i.e. how big is a pixel) in microns
    z : float
        distance to propagate, microns

    Returns
    -------
    numpy.ndarray
        transfer function of free space
    """
    ky, kx = (fft.fftfreq(s, dx) for s in shape)
    ky = np.broadcast_to(ky, shape).swapaxes(0, 1)
    kx = np.broadcast_to(kx, shape)

    coef = np.pi * wvl * z
    transfer_function = np.exp(-1j * coef * (kx**2 + ky**2))
    return transfer_function

def _angular_spectrum_prop(field, transfer_function):
    """Propagate a field using the angular spectrum method

    Parameters
    ----------
    field : numpy.ndarray
        field to propagate
    transfer_function : numpy.ndarray
        transfer function of free space, call _angular_spectrum_transfer_function()

    Returns
    -------
    numpy.ndarray
        propagated field
    """
    # this code is copied from prysm with some modification
    forward = fft.fft2(field)
    return fft.ifft2(forward*transfer_function)


def ft_fwd(x):
    """ 'focus' operator, wrapper for numpy fft that conserves energy

    Parameters
    ----------
    x : numpy.ndarray
        field to focus

    Returns
    -------
    numpy.ndarray
        focused field
    """
    return fft.ifftshift(fft.fft2(fft.fftshift(x), norm='ortho'))


def ft_rev(x):
    """ 'unfocus' operator, wrapper for numpy fft that conserves energy

    Parameters
    ----------
    x : numpy.ndarray
        field to unfocus

    Returns
    -------
    numpy.ndarray
        unfocused field
    """
    return fft.ifftshift(fft.ifft2(fft.fftshift(x), norm='ortho'))


def convolve_2d(x, kernel, normalize=True):
    """Convolves x with kernel using ffts

    Parameters
    ----------
    x: ndarray
        signal to convolve
    kernel: ndarray
        kernel for convolution
    normalize: bool
        whether to maintain the energy of x in the
        convolution, defaults to True

    Returns
    -------
    ndarray
        convolved signal

    """

    spectrum = ft_fwd(x)
    kernelft = ft_fwd(kernel)
    spectral_weights = ft_fwd(np.ones_like(x))

    convolved = ft_rev(spectrum * kernelft)
    weights = ft_rev(spectral_weights * kernelft)

    if normalize:
        convolved /= weights
        convolved = np.abs(convolved)
        convolved = np.clip(convolved, 0, 1)

    return convolved

