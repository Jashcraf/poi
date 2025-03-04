from prysm.coordinates import make_xy_grid, cart_to_polar
from prysm.segmented import (
    CompositeHexagonalAperture,
    VERTEX_TO_VERTEX_TO_FLAT_TO_FLAT
)
from prysm.propagation import (
    focus_fixed_sampling,
    focus_fixed_sampling_backprop,
    unfocus_fixed_sampling,
    unfocus_fixed_sampling_backprop
)
from prysm.x.dm import DM

from prysm.mathops import np

FTF_KECK = 1800 * VERTEX_TO_VERTEX_TO_FLAT_TO_FLAT

KECK_CORO_IWA_AS = [
    0.1, 0.15, 0.2, 0.3, 0.4,
    0.6, 0.8, 1., 1.5, 2.
]

KECK_CORO_IWA_LAMD = np.asarray(KECK_CORO_IWA_AS) / 206265
KECK_CORO_IWA_LAMD *= 10.95 / 1.65e-6

class TelescopeModel:
    """Base class for constructing telescope models
    """

    def __init__(self, Npup, Nimg, dx_pup, dx_img, center_wavelength,
                 bandwidth, epd, efl, num_wavelengths, IWA, OWA,
                 LS_inner, LS_outer):
        """Initializes a base telescope model

        Parameters
        ----------
        Npup : int
            number of samples to use across the pupil
        Nimg : int
            number of samples to use across the image
        dx_pup : float
            pixelscale of a sample in the pupil plane, milimeters
        dx_img : float
            pixelscale of a sample in the image plane, microns
        center_wavelength : float
            center wavelength to observe in microns
        bandwidth : float
            bandwidth of observation centered on center_wavelength, 
            as a percentage
        epd : float
            entrance pupil diameter in milimeters, typically the primary
            mirror diameter
        efl : float
            effective focal length in milimeters
        num_wavelengths : int
            number of wavelengths with which to sample the bandwidth
        IWA : float
            inner working angle of a lyot coronagraph in lambda / D, 
            units of the center wavelength
        OWA : float
            outer working angle of a lyot coronagraph in lambda / D,
            units of the center wavelength. If None, returns a occulting spot
            focal plane mask
        LS_inner : float
            Fractional inner circular diameter of the lyot stop obscuration. If None,
            returns a circular Lyot stop
        LS_outer : float
            Fractional outer circular diameter of the lyot stop obscuration. 
            If LS_outer=1, then it returns a circular pupil with the same diameter
            as the `epd`. LS=0.8 will return a lyot stop with 80% the radius.
        """

        self.Npup = Npup
        self.Nimg = Nimg
        self.dx_pup = dx_pup
        self.dx_img = dx_img
        self.center_wavelength = center_wavelength
        self.bandwidth = bandwidth / 100 # percent to fraciton converison
        self.epd = epd
        self.efl = efl
        self.nwvls = num_wavelengths
        self.IWA = IWA
        self.OWA = OWA
        self.LS_inner = LS_inner
        self.LS_outer = LS_outer
        self.um_to_lamD = 1e-3 / (self.efl) * self.epd / (self.center_wavelength * 1e-3)

        if self.nwvls > 1: 
            self.wavelengths = np.linspace((1 - self.bandwidth/2) * center_wavelength,
                                           (1 + self.bandwidth/2) * center_wavelength,
                                            num_wavelengths)
        else:
            self.wavelengths = [self.center_wavelength]

        # configure the lyot coronagraph
        x, y = make_xy_grid(self.Npup, dx=self.dx_pup)
        r, _ = cart_to_polar(x, y)
        r /= self.epd / 2
        u, v = make_xy_grid(self.Nimg, dx=self.dx_img * self.um_to_lamD)
        rho, _ = cart_to_polar(u, v)

        # set up lyot stop
        self.lyot_stop = np.zeros_like(x)
        self.lyot_stop[r < self.LS_outer] = 1

        if self.LS_inner is not None:
            self.lyot_stop[r < self.LS_inner] = 0

        # set up focal plane mask
        self.fpm = np.ones_like(u)
        self.fpm[rho < self.IWA] = 0

        if self.OWA is not None:
            self.fpm[rho > self.OWA] = 0


class KeckTelescope(TelescopeModel):
    """Construct an instance of a Keck II PSF Simulator
    """

    def __init__(self, center_wavelength, bandwidth,
                 epd=10.95e3, efl=17.5e3, num_wavelengths=5,
                 Npup=1024, Nimg=128, dx_pup=10.95e3/1024, dx_img=1,
                 IWA=3, OWA=None, LS_inner=0.2, LS_outer=0.8):
        """Initializes a Keck-like telescope model
        
        Notes
        -----
        segment perscription taken from section 3.1.1 of
        https://www2.keck.hawaii.edu/observing/kecktelgde/ktelinstupdate.pdf
        """

        super().__init__(Npup, Nimg, dx_pup, dx_img, center_wavelength,
                         bandwidth, epd, efl, num_wavelengths, IWA, OWA,
                         LS_inner, LS_outer)

        # Draw the Keck Aperture
        x, y = make_xy_grid(self.Npup, dx=self.dx_pup)
        cha = CompositeHexagonalAperture(x, y, 
                                         rings=3,
                                         segment_diameter=FTF_KECK,
                                         segment_separation=5,
                                         exclude=(0,))
        
        self._composite_aperture = cha

    def get_entrance_pupil(self):
        """Returns entrance pupil array

        Returns
        -------
        ndarray
            2D array containing the entrance pupil
        """
        return self._composite_aperture.amp
    
    def get_direct_image(self, wfe=None):
        """Simulate a (generally) polychromatic image

        Parameters
        ----------
        wfe : ndarray, optional
            wavefront error to apply to the entrance pupil in microns,
            by default None, which applies no wavefront error

        Returns
        -------
        ndarray
            focal plane
        """

        if wfe is None:
            wfe = 0

        for i, wvl in enumerate(self.wavelengths):

            # construct wfe phasor
            wfe_phasor = np.exp(1j * 2 * np.pi / wvl * wfe)

            focus = focus_fixed_sampling(self.get_entrance_pupil() * wfe_phasor,
                                         input_dx=self.dx_pup, 
                                         prop_dist=self.efl,
                                         wavelength=wvl, # microns
                                         output_dx=self.dx_img, # microns
                                         output_samples=self.Nimg)
            
            intensity = np.abs(focus)**2 / len(self.wavelengths)

            if i == 0:
                psf = intensity
            else:
                psf += intensity

        return psf
    
    def get_coronagraph_image(self, wfe=None, include_fpm=True):
        """Simulate a (generally) coronagraphic image. Currently supports
        the Lyot coronagraph mode, which is initialized by default

        Parameters
        ----------
        wfe : ndarray, optional
            wavefront error to apply to the entrance pupil in microns,
            by default None, which applies no wavefront error
        include_fpm : bool, optional
            Whether to include the focal plane mask in the simulation,
            by default True, which leaves it in. 

        Returns
        -------
        ndarray
            coronagraphic image plane
        """

        if wfe is None:
            wfe = 0

        for i, wvl in enumerate(self.wavelengths):

            # construct wfe phasor
            wfe_phasor = np.exp(1j * 2 * np.pi / wvl * wfe)

            # go to psf
            focus = focus_fixed_sampling(self.get_entrance_pupil() * wfe_phasor,
                                         input_dx=self.dx_pup,
                                         prop_dist=self.efl,
                                         wavelength=wvl, # microns
                                         output_dx=self.dx_img, # microns
                                         output_samples=self.Nimg)
            
            # apply focal plane mask
            # optional for Normalized intensity calculations
            if include_fpm:
                focus *= self.fpm

            # propagate
            lyot = unfocus_fixed_sampling(focus,
                                          input_dx=self.dx_img,
                                          prop_dist=self.efl,
                                          wavelength=wvl, # microns
                                          output_dx=self.dx_pup, # microns
                                          output_samples=self.Npup)

            # apply lyot stop
            lyot *= self.lyot_stop

            # go to coro image
            coro = focus_fixed_sampling(lyot,
                                        input_dx=self.dx_pup,
                                        prop_dist=self.efl,
                                        wavelength=wvl, # microns
                                        output_dx=self.dx_img, # microns
                                        output_samples=self.Nimg)
            
            intensity = np.abs(coro)**2 / len(self.wavelengths)

            if i == 0:
                psf = intensity
            else:
                psf += intensity

        return psf








        
