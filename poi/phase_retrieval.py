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
from .poi_math import broadcast_kron

A = np.array([
    [1, 0, 0, 1],
    [1, 0, 0, -1],
    [0, 1, 1, 0],
    [0, 1j, -1j, 0]
    ])


"""Largely taken from dydgug.vappid.VAPPOptimizer2, with minor modifications to support focus diversity"""
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
    
class PZPhaseRetireval:
    def __init__(self, amp, amp_dx, efl, wvl, basis, target, img_dx,
                 defocus_waves=0, initial_phase=None, stokes=np.array([1.,0.,0.,0.])):
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
        self.stokes = stokes

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
            
            r_xx = x[0*NMODES : 1*NMODES]
            r_xy = x[1*NMODES : 2*NMODES]
            r_yx = x[2*NMODES : 3*NMODES]
            r_yy = x[3*NMODES : 4*NMODES]
            
            i_xx = x[4*NMODES : 5*NMODES]
            i_xy = x[5*NMODES : 6*NMODES]
            i_yx = x[6*NMODES : 7*NMODES]
            i_yy = x[7*NMODES : 8*NMODES]

            c_xx = r_xx + 1j*i_xx
            c_xy = r_xy + 1j*i_xy
            c_yx = r_yx + 1j*i_yx
            c_yy = r_yy + 1j*i_yy
            
            Jxx = np.tensordot(self.basis, c_xx, axes=(0,0))
            Jxy = np.tensordot(self.basis, c_xy, axes=(0,0))
            Jyx = np.tensordot(self.basis, c_yx, axes=(0,0))
            Jyy = np.tensordot(self.basis, c_yy, axes=(0,0))
            Jones = np.array([
                [Jxx, Jxy],
                [Jyx, Jyy]
            ])
            Jones = np.moveaxis(Jones, -1, 0)
            ARM = np.zeros_like(Jones)        
        # Apply polarization diversity with waveplate

        # TODO: Check if this is a minus sign instead
        for i in range(2):
            for j in range(2):

                g = Jones[..., i, j] * np.exp(-1j * self.defocus_diversity)
                G = focus_fixed_sampling(
                    wavefunction=g,
                    input_dx=self.amp_dx,
                    prop_dist = self.efl,
                    wavelength=self.wvl,
                    output_dx=self.img_dx,
                    output_samples=self.D.shape,
                    shift=(0, 0),
                    method='mdft')

                ARM[..., i, j] = G
        
        # Convert to Mueller Matrix
        MPSM = A @ broadcast_kron(ARM, np.conj(ARM)) @ np.linalg.inv(A)

        # Dot with stokes in
        I = MPSM @ self.stokes
        E = np.sum((I - self.D)**2)
        
        self.ARM = ARM
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

        Mbar = Ibar * self.stokes
        
        # Construct the gradient backpropagation matrix from the ARM
        A11 = self.ARM[..., 0, 0].conj()
        A12 = self.ARM[..., 0, 1].conj()
        A21 = self.ARM[..., 1, 0].conj()
        A22 = self.ARM[..., 1, 1].conj()
        
        # TODO: Check on the shape of Abar
        Abar = np.array([
            [A11, A11, A12, -1j * A12],
            [A12, -A12, A11, 1j * A11],
            [A21, A21, A22, -1j * A22],
            [A22, -A22, A12, 1j * A12]
        ])

        vec_Jbar = Abar @ Mbar
        Jbar = vec_Jbar.reshape([*vec_Jbar.shape[:-1], 2, 2])
        hbar = np.zeros_like(Jbar)

        for i in range(2):
            for j in range(2):
                
                Gbar = Jbar[..., i, j]

                gbar = focus_fixed_sampling_backprop(
                    wavefunction=Gbar,
                    input_dx=self.amp_dx,
                    prop_dist = self.efl,
                    wavelength=self.wvl,
                    output_dx=self.img_dx,
                    output_samples=self.phs.shape,
                    shift=(0, 0),
                    method='mdft')
                
                hbar[..., i, j] = gbar * self.amp
        
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
