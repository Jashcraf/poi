from prysm.mathops import np

from prysm.propagation import (
    focus_fixed_sampling,
    focus_fixed_sampling_backprop
)

from prysm.coordinates import (
    make_xy_grid,
    cart_to_polar
)

from prysm.polynomials import (
    hopkins
)

from prysm.x.polarization import linear_polarizer, quarter_wave_plate
from .propagation import _angular_spectrum_prop, _angular_spectrum_transfer_function
from .processing import mean_squared_error

from warnings import warn

"""Largely taken from dydgug.vappid.VAPPOptimizer2, with minor modifications to support focus diversity"""

def bias_and_gain_invariant_error(I, D, mask):  # NOQA
    """taken from prysm.x.optym by Brandon Dube

    Bias and gain invariant error.

    This cost function computes internal least mean squares estimates of the
    overall bias (DC pedestal) and gain between the signal I and D.  This
    objective is useful when the overall signal level is ambiguous in phase
    retrieval type problems, and can significantly help stabilize the
    optimization process.

    See also: mean_square_error

    Parameters
    ----------
    I : ndarray
        'intensity' or model data, any float dtype, any shape
    D : ndarray
        'data' or true mesaurement to be matched, any float dtype, any shape
    mask : ndarray
        logical array with elements to keep (True) or exclude (False)

    Returns
    -------
    float, ndarray
        cost, dcost/dI


    """
    if mask is not None:
        grad = np.zeros_like(I)
        I = I[mask]  # NOQA
        D = D[mask]

    Ihat = I - I.mean()  # zero mean
    Dhat = D - D.mean()

    N = I.size

    num = (Ihat*Dhat).sum()
    den = (Ihat*Ihat).sum()
    alpha = num/den

    alphaI = alpha*I

    beta = (D-alphaI)/N

    R = 1/((D*D).sum())
    raw_err = (alphaI + beta) - D
    err = R*(raw_err*raw_err).sum()
    # 2 is from raw_err squared
    # R is multiplied and not part of the differentiation, pass-through
    # likewise with alpha
    grad = 2*R*alpha*raw_err

    if mask is not None:
        grad2 = np.zeros(mask.shape, dtype=I.dtype)
        grad2[mask] = grad
        grad = grad2

    return err, grad

class ADPhaseRetireval:
    def __init__(self, amp, amp_dx, efl, wvl, basis, target, img_dx, defocus_waves=0, initial_phase=None):
        if initial_phase is None:
            phs = np.zeros(amp.shape, dtype=float)
        else:
            phs = initial_phase

        self.amp = amp
        self.amp_select = self.amp > 1e-9
        self.amp_dx = amp_dx
        self.epd = amp.shape[0] * amp_dx
        self.efl = efl
        self.wvl = wvl
        self.basis = basis
        self.img_dx = img_dx
        self.D = target
        self.phs = phs
        self.zonal = False
        self.defocus = defocus_waves

        # configure the defocus polynomial
        x, y = make_xy_grid(amp.shape[0], diameter=self.epd)
        r, t = cart_to_polar(x, y)
        r_z = r / (self.epd / 2)
        self.defocus_polynomial = hopkins(0, 2, 0, r_z, t, 0)
        self.defocus_aberration = 2 * np.pi * self.defocus_polynomial * self.defocus * self.amp
        self.cost = []

    def set_optimization_method(self, zonal=False):
        self.zonal = zonal

    def update(self, x):
        if not self.zonal:
            phs = np.tensordot(self.basis, x, axes=(0,0))
        else:
            phs = np.zeros(self.amp.shape, dtype=float)
            phs[self.amp_select] = x

        W = (2 * np.pi / self.wvl) * phs

        # TODO: Check if this is a minus sign instead
        W -= self.defocus_aberration
        g = self.amp * np.exp(1j * W)
        G = focus_fixed_sampling(
            wavefunction=g,
            input_dx=self.amp_dx,
            prop_dist = self.efl,
            wavelength=self.wvl,
            output_dx=self.img_dx,
            output_samples=self.D.shape,
            shift=(0, 0),
            method='mdft')
        I = np.abs(G)**2
        
        # E = np.sum((I - self.D)**2)
        err, grad = bias_and_gain_invariant_error(I, self.D)
        self.Ibar = grad 
        
        self.E = err
        self.phs = phs
        self.W = W
        self.g = g
        self.G = G
        self.I = I
        self.E = E
        return

    def fwd(self, x):
        self.update(x)
        return self.E

    def rev(self, x):
        self.update(x)
        #Ibar = 2*(self.I - self.D)
        Gbar = 2 * self.Ibar * self.G
        gbar = focus_fixed_sampling_backprop(
            wavefunction=Gbar,
            input_dx=self.amp_dx,
            prop_dist = self.efl,
            wavelength=self.wvl,
            output_dx=self.img_dx,
            output_samples=self.phs.shape,
            shift=(0, 0),
            method='mdft')

        Wbar = 2 * np.pi / self.wvl * np.imag(gbar * np.conj(self.g))
        if not self.zonal:
            abar = np.tensordot(self.basis, Wbar)

        self.Ibar = Ibar
        self.Gbar = Gbar
        self.gbar = gbar
        self.Wbar = Wbar

        if not self.zonal:
            self.abar = abar
            return self.abar
        else:
            return self.Wbar[self.amp_select]

    def fg(self, x):
        g = self.rev(x)
        f = self.E
        self.cost.append(f)
        return f, g


class PZPhaseRetireval:
    """Class to perform jones pupil phase retrieval

    the shape of x should be 4 x len(basis), but minimize doesn't like that
    so we will need to reshape. This generally supports an upstream retarder and
    downstream polarizer, but the retardance and diattenuation can be set to
    make these an identity matrix.

    Parameters
    ----------
    amp : numpy.ndarray
        amplitude of the pupil function
    amp_dx : float
        pixel scale of the pupil function in milimeters
    efl : float
        effective focal length of the system in miilimeters
    wvl : float
        wavelength of light in microns
    basis : numpy.ndarray
        basis set to use for the phase (e.g. Zernikes, Hexikes, etc)
    target : numpy.ndarray
        target PSF intensity to fit to
    img_dx : float
        pixel scale of the target PSF in microns
    defocus_waves : float, optional
        amount of defocus diversity to use in waves, by default 0
    retarder_angle : float, optional
        angle of the retarder fast axis in radians, by default 0
    polarizer_angle : float, optional
        angle of the polarizer transmission axis in radians, by default 0
    retardance : float, optional
        retardance of the upstream waveplate in radians, by default np.pi/4
    Tmin : float, optional
        minimum transmission of the polarizer, by default 0. Related to the
        diattenuation.
    initial_phase : numpy.ndarray, optional
        initial phase guess, by default None, which sets a random phase guess.

    """
    def __init__(self, amp, amp_dx, efl, wvl, basis, target, img_dx,
                 defocus_waves=0, retarder_angle=0, polarizer_angle=0,
                 retardance=np.pi/4, Tmin=0, initial_phase=None):
        if initial_phase is None:
            phs = np.zeros(amp.shape, dtype=float)

        self.amp = amp
        self.amp_select = self.amp > 1e-9
        self.amp_dx = amp_dx
        self.epd = amp.shape[0] * amp_dx
        self.efl = efl
        self.wvl = wvl
        self.basis = basis
        self.img_dx = img_dx
        self.D = target
        self.phs = phs
        self.zonal = False
        self.defocus = defocus_waves
        self.retarder_angle = retarder_angle
        self.polarizer_angle = polarizer_angle
        self.lenbasis = len(self.basis)

        # configure the defocus polynomial
        x, y = make_xy_grid(amp.shape[0], diameter=self.epd)
        r, t = cart_to_polar(x, y)
        r_z = r / (self.epd / 2)
        self.defocus_polynomial = hopkins(0, 2, 0, r_z, t, 0)
        self.defocus_aberration = 2 * np.pi * self.defocus_polynomial * self.defocus * self.amp
        self.cost = []

        # configure polarization diversity
        self.R = quarter_wave_plate(theta=self.retarder_angle, shape=amp.shape)
        self.P = linear_polarizer(theta=self.polarizer_angle, shape=amp.shape)
        warn("NOTE: Retardance and Diattenuation are not yet implemented in the phase retrieval.")

    def set_optimization_method(self, zonal=False):
        self.zonal = zonal

    def update(self, x):

        # reshape x
        x = x.reshape([self.lenbasis, 4])

        if not self.zonal:
            phs = np.tensordot(self.basis, x, axes=(0, 0))

        else:
            phs = np.zeros(self.amp.shape, dtype=float)
            phs[self.amp_select] = x

        W = (2 * np.pi / self.wvl) * phs

        # TODO: Check if this is a minus sign instead
        W -= self.defocus_aberration
        g = self.amp * np.exp(1j * W)
        k = g @ self.R
        G = focus_fixed_sampling(
            wavefunction=k,
            input_dx=self.amp_dx,
            prop_dist = self.efl,
            wavelength=self.wvl,
            output_dx=self.img_dx,
            output_samples=self.D.shape,
            shift=(0, 0),
            method='mdft')
        I = np.abs(G)**2
        E = np.sum((I - self.D)**2)
        self.phs = phs
        self.W = W
        self.g = g
        self.G = G
        self.I = I
        self.E = E
        return

    def fwd(self, x):
        self.update(x)
        return self.E

    def rev(self, x):
        self.update(x)
        Ibar = 2*(self.I - self.D)
        Gbar = 2 * Ibar * self.G
        gbar = focus_fixed_sampling_backprop(
            wavefunction=Gbar,
            input_dx=self.amp_dx,
            prop_dist = self.efl,
            wavelength=self.wvl,
            output_dx=self.img_dx,
            output_samples=self.phs.shape,
            shift=(0, 0),
            method='mdft')

        Wbar = 2 * np.pi / self.wvl * np.imag(gbar * np.conj(self.g))
        if not self.zonal:
            abar = np.tensordot(self.basis, Wbar)

        self.Ibar = Ibar
        self.Gbar = Gbar
        self.gbar = gbar
        self.Wbar = Wbar

        if not self.zonal:
            self.abar = abar
            return self.abar
        else:
            return self.Wbar[self.amp_select]

    def fg(self, x):
        g = self.rev(x)
        f = self.E
        self.cost.append(f)
        return f, g


class ParallelADPhaseRetrieval:

    def __init__(self, optlist):

        self.optlist = optlist
        self.f = 0
        self.g = 0
        self.cost = []

    def refresh(self):
        self.f = 0
        self.g = 0

    def fg(self, x):

        # reset the f, g values
        self.refresh()

        # just sum them
        for opt in self.optlist:
            f, g = opt.fg(x)
            self.f += f
            self.g += g

        self.cost.append(self.f)

        return self.f, self.g


class FocusDiversePhaseRetrieval:
    """Focus Diversity Phase Retrieval using Gerchberg-Saxton-like iteration.

    Algorithm inspired by Misel's two-psf algorithm [1], generalized to N psfs
    - [1] D L Misell 1973 J. Phys. D: Appl. Phys. 6 2200
    """

    def __init__(self,psflist,wvl,dxs,defocus_positions,phase_guess=None):
        """Phase Retrieval Iterator using Focus Diversity for N defocus positions

        Parameters
        ----------
        psflist : list of numpy.ndarrays of the same shape
            length N list of numpy.ndarrays that contain the defocused PSF data. Must be of the same pixel scale
            and array size
        wvl : float
            wavelength of light in microns
        dxs : float
            pixel scale of the arrays in psflist in microns
        defocus_positions : list of floats
            defocus positions in microns
        phase_guess : numpy.ndarray, optional
            phase guess of the desired pupil sampling, by default None
        """

        # catch some common mistakes
        assert len(defocus_positions) == len(dxs), f"defocus_positions and dxs should have the same length, got {len(defocus_positions)} and {len(dxs)}"
        assert (len(psflist) == len(dxs)+1) and (len(psflist) == len(defocus_positions)+1), f"psflist should be one element longer than dxs and defocus_positions, got {len(psflist)}"

        try:
            if phase_guess is None:
                phase_guess = np.random.rand(*psflist[0].shape)

            self.absFlist = []
            self.mse_denom = []

            # TODO: Throw a try-except

            # Create the object domain data in field units
            for psf in psflist:
                self.absFlist.append(np.fft.ifftshift(np.sqrt(psf)))
                self.mse_denom.append(np.sum(psf))

            # Begin with a guess using the first PSF
            phase_guess = np.fft.ifftshift(phase_guess)
            self.G0 = self.absFlist[0] * np.exp(1j*phase_guess)

            # pre-compute transfer functions, lists of kernels
            self.forward_prop = []
            self.backward_prop = []
            self.cost_functions = [] # will be a list of lists
            for dz,dx in zip(defocus_positions,dxs):
                self.forward_prop.append(_angular_spectrum_transfer_function(psflist[0].shape,wvl,dx,dz)) # there was a 1e-3 factor here
                self.backward_prop.append(_angular_spectrum_transfer_function(psflist[0].shape,wvl,dx,-dz))
                self.cost_functions.append([])

            self.iter = 0

        except Exception as e:
            self.log.critical(f'Error in initializing iterator: \n {e}')

    def step(self):
        """use Misel's algorithm to perform an iteration between image space and the fourier plane

        Returns
        -------
        G0primeprime
            updated estimate of the image plane electric field
        """

        for i,(fwd,rev,absF1,mse_denom) in enumerate(zip(self.forward_prop,self.backward_prop,self.absFlist[1:],self.mse_denom)):

            G1 = _angular_spectrum_prop(self.G0,fwd)
            phs_G1 = np.angle(G1)
            G1prime = absF1 * np.exp(1j*phs_G1)
            G0prime = _angular_spectrum_prop(G1prime,rev)
            phs_G0prime = np.angle(G0prime)
            # G0primeprime = self.absFlist[0] * np.exp(1j*phs_G0prime)
            G0primeprime = self.absFlist[0] * np.exp(1j*phs_G0prime)

            # remember to update the phase guess for PSF
            self.G0 = G0primeprime
            self.cost_functions[i].append(mean_squared_error(np.abs(G0prime),self.absFlist[0],norm=mse_denom))
            self.iter += 1

        # return pupil_estimate
        # pupil_estimate = np.fft.ifftshift(np.fft.ifft2(G0primeprime))

        return np.fft.fftshift(G0primeprime)
