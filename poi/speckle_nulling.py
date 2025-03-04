from prysm.mathops import np
from prysm.coordinates import make_xy_grid, cart_to_polar
from .modes import fourier_modes_sequence

import matplotlib.pyplot as plt

def find_max_index(array):
    index = np.unravel_index(array.argmax(), array.shape)
    return index


def make_quadrant_masks(image):

    x, y = make_xy_grid(image.shape, diameter=1)

    # set up diagonals
    true_mask = np.zeros_like(image)
    q1 = np.copy(true_mask)
    q2 = np.copy(true_mask)
    q3 = np.copy(true_mask)
    q4 = np.copy(true_mask)

    q1[(x > y) & (-x < y)] = 1  # East
    q2[(y > -x) & (x < y)] = 1  # South
    q3[(y > x) & (x < -y)] = 1  # West
    q4[(x > y) & (x < -y)] = 1  # North

    return [q1, q2, q3, q4]


def find_cardinal_points(image):

    quads = make_quadrant_masks(image)
    coords = []

    for q in quads:

        row, col = find_max_index(q * image)
        coords.append((row, col))

    return coords


def find_image_center(image):

    # returns East, South, West, North
    coords = find_cardinal_points(image)
    row_center = 0
    col_center = 0

    for coord in coords:
        row_center += coord[0] / len(coords)
        col_center += coord[1] / len(coords)

    return row_center, col_center


def compute_pixel_scale_lamd(image, waffle_frequency=5):

    rowc, colc = find_image_center(image)
    coords = find_cardinal_points(image)
    row_differences = []
    col_differences = []

    for coord in coords:
        row_differences.append((coord[0] - rowc)**2)
        col_differences.append((coord[1] - colc)**2)

    # average the pixelscale
    separations = np.sqrt(row_differences + col_differences)
    mean_separation = np.mean(separations) / waffle_frequency / 2 * np.pi

    return mean_separation


class SpeckleNulling:

    def __init__(self, propagation, dx_img, epd, efl, wvl, dm, dh, ref_contrast=1, nsteps=4):

        self.fwd = propagation
        self.dm = dm
        self.dx_img = dx_img
        self.epd = epd
        self.efl = efl
        self.wvl = wvl
        self.image_dx_lamD = self.dx_img * 1e-6 / (self.efl * 1e-3) * (self.epd * 1e-3) / (self.wvl * 1e-6)
        self.dh = dh
        self.kvec = 2 * np.pi / wvl
        self.images = []
        self.mean_in_dh = []
        self.dm_surface = []
        self.ref_contrast = ref_contrast
        self.nsteps = 4
        self.nact = self.dm.Nact

    def get_image_center(self, mode_amplitude=1e-1):
        """Takes an image at the frequency of the DM by applying a Waffle mode
        """
        waffle = np.ones_like(self.dm.actuators) * mode_amplitude
        for i in range(self.dm.actuators.shape[0]):
            for j in range(self.dm.actuators.shape[1]):

                if i%2 != 0 or j%2 != 0:
                    waffle[i,j] = 0

        self.dm.actuators[:] += waffle
        waffle_surface = self.dm.render()
        self.waffle_image = self.fwd(self.dm.render(wfe=True))
        self.dm.actuators[:] -= waffle

        return self.waffle_image, waffle_surface

    def get_calibrate_img_pixelscale(self, waffle_frequency=5):
        image = self.waffle_image
        rowc, colc = find_image_center(image)
        coords = find_cardinal_points(image)
        row_differences = []
        col_differences = []

        for coord in coords:
            row_differences.append((coord[0] - rowc)**2)
            col_differences.append((coord[1] - colc)**2)

        # average the pixelscale
        separations = np.sqrt(row_differences + col_differences)
        mean_separation = np.mean(separations) / waffle_frequency / 2 * np.pi

    def step(self, return_probe_images=False, probe_amplitude=1e-2):

        if not hasattr(self, "waffle_image"):
            print("No waffle image detected, taking waffle image for image center")
            self.get_image_center()

        if len(self.images) < 1:
            print("Taking starter image at position zero")
            self.images.append(self.fwd(self.dm.render(wfe=True)))

         # the probe offsets
        if self.nsteps == 4:
            probe_phases = [0, np.pi/2, np.pi, 3 * np.pi / 2]
        else:
            raise NotImplementedError

        probe_images = []

        # find location of max speckle in dark_hole
        img = self.fwd(self.dm.render(wfe=True)) * self.dh
        img_max = img.max() / self.ref_contrast
        rowmax, colmax = find_max_index(img)

        # convert to a frequency in the fourier domain
        rowc, colc = find_image_center(self.waffle_image)

        row_freq = (rowmax - rowc) * self.image_dx_lamD
        col_freq = (colmax - colc) * self.image_dx_lamD

        # construct the cosinusoid
        udm, vdm = make_xy_grid(self.nact, diameter=1)

        for probe in probe_phases:
            probe_speckle = np.cos(2 * np.pi * (udm * col_freq + vdm * row_freq) + probe) * probe_amplitude
            self.dm.actuators[:] += probe_speckle
            probe_image = self.fwd(self.dm.render(wfe=True)) / self.ref_contrast
            probe_images.append(probe_image)
            self.dm.actuators[:] -= probe_speckle

        # phase unwrapping algorithm
        I1 = probe_images[0][rowmax, colmax]
        I2 = probe_images[1][rowmax, colmax]
        I3 = probe_images[2][rowmax, colmax]
        I4 = probe_images[3][rowmax, colmax]

        real = I1 - I3
        imag = I4 - I2
        phase = np.arctan2(imag, real)
        gain = 0.5

        # amplitude from real/imag
        # amplitude = np.sqrt(real**2 + imag**2) * gain

        # amplitude from implicit conversion
        # This method was taken from Luci Lebilloux's HICAT paper
        probe_intensity = I1
        speckle_intensity = img_max
        amplitude = gain * probe_amplitude * np.sqrt(img_max / I1)

        # construct sinusoid with pi phase offset to cancel
        phase += np.pi # to cancel
        probe_speckle = np.cos(2 * np.pi * (udm * col_freq + vdm * row_freq) - phase)
        probe_speckle *= amplitude

        # apply the correction
        self.dm.actuators[:] += probe_speckle

        # return an image
        img_corrected = self.fwd(self.dm.render(wfe=True))
        self.images.append(img_corrected)

        if return_probe_images:
            return img_corrected, probe_images
        else:
            return img_corrected


class SpeckleAreaNulling:

    def __init__(self, propagation, dx_img, epd, efl, wvl, dm, IWA, OWA,
                 ref_contrast=1, edge=None, angular_range=[-90, 90]):
        """Instance of Speckle Area Nulling

        Parameters
        ----------
        propagation : callable
            function that, when called, returns an intensity image
        dx_img : float
            Image pixel size in microns
        epd : float
            Entrance pupil diameter in milimeters
        efl : float
            Effective focal length in milimeters
        wvl : float
            Wavelength in microns
        dm : DM
            Instance of DM class
        IWA : float
            Inner working angle in lambda / D
        OWA : float
            Outer working angle in lambda / D
        ref_contrast : float, optional
            Reference contrast, used to divide intensity images to get in units of
            normalized intensity, by default 1, which returns values in units of intensity
        edge : float, optional
            A knife-edge limit to the dark hole region in lambda / D, by default None, which
            returns a full dark hole
        angular_range : list, optional
            Angular range in degrees for the dark hole region, by default [-90, 90]

        Returns
        -------
        SpeckleAreaNulling
            Instance of SpeckleAreaNulling class
        """

        self.fwd = propagation
        self.dm = dm
        self.dx_img = dx_img
        self.epd = epd
        self.efl = efl
        self.wvl = wvl
        self.image_dx_lamD = self.dx_img * 1e-3 / (self.efl) * (self.epd) / (self.wvl * 1e-3)
        self.IWA = IWA
        self.OWA = OWA
        self.kvec = 2 * np.pi / wvl
        self.images = []
        self.mean_in_dh = []
        self.dm_surface = []
        self.ref_contrast = ref_contrast
        self.nact = self.dm.Nact
        self.edge = edge

        # construct a dark hole
        self.Nimg = self.fwd().shape[0]
        self.x, self.y = make_xy_grid(self.Nimg, dx=self.dx_img * 1e-3)

        self.u, self.v = make_xy_grid(self.Nimg, dx=self.image_dx_lamD)
        r, t = cart_to_polar(self.u, self.v)

        self.dh = np.zeros([self.Nimg, self.Nimg])
        self.dh[r < self.OWA] = 1.
        self.dh[r < self.IWA] = 0.

        self.dh[t < angular_range[0]] = 1.
        self.dh[t > angular_range[1]] = 1.

        if self.edge is not None:
            self.dh[self.u < self.edge] = 0.

        # Bulid up the fourier modes in the dark hole region
        kfreq = self.kvec / (self.efl * 1e-3)  # convert wavelength to milimeters
        xfreq = (kfreq * self.x)[self.dh == 1]
        yfreq = kfreq * self.y[self.dh == 1]

        # Make the DM shapes
        xdm, ydm = make_xy_grid(self.nact, dx=self.dm.act_pitch * 1e-1)
        arg = xfreq[..., None, None] * xdm + yfreq[..., None, None] * ydm
        self.cos_modes = np.cos(arg)
        self.sin_modes = np.sin(arg)
        
        xdm /= 8
        ydm /= 8

        V1sum = np.sum(self.sin_modes, axis=0)
        V2sum = np.sum(self.cos_modes, axis=0)

        V1norm = np.abs(V1sum).max()
        V2norm = np.abs(V2sum).max()
        V1sum /= V1norm
        V2sum /= V2norm

        self.sin_probe = V1sum
        self.cos_probe = V2sum
        
        self.sin_modes /= V1norm * 10
        self.cos_modes /= V2norm * 10

    def step(self, regularization=5e-4):
        """Advance the algorithm one iteration

        Parameters
        ----------
        regularization : float
            Regularization parameter to minimize the inversion of small signals.
            This is roughly analogous to the change in contrast that you wish to
            ignore.
        """

        # Starting image acquisition
        I0 = self.fwd() / self.ref_contrast

        # Four probe steps
        for probe in [-self.sin_probe, self.sin_probe, -self.cos_probe, self.cos_probe]:
            self.dm.actuators[:] += probe
            I = self.fwd(self.dm.render(wfe=True)) / self.ref_contrast
            self.dm.actuators[:] -= probe
            self.images.append(I)


        Im1 = self.images[-4]  # minus sin probe
        Ip1 = self.images[-3]  # plus sin probe
        Im2 = self.images[-2]  # minus cos probe
        Ip2 = self.images[-1]  # plus cos probe

        # Compute the relevant quantities
        dE1 = (Ip1 - Im1) / 4
        dE2 = (Ip2 - Im2) / 4
        dE1sq = (Ip1 + Im1 - 2*I0) / 2
        dE2sq = (Ip2 + Im2 - 2*I0) / 2

        # Regularized sin / cosine coefficients
        sin_coeffs = dE1 / (dE1sq + regularization)
        cos_coeffs = dE2 / (dE2sq + regularization)
        

        # Just the ones in the dark hole
        sin_coeffs = sin_coeffs[self.dh==1, None, None]
        cos_coeffs = cos_coeffs[self.dh==1, None, None]

        # apply the correction
        correction = sin_coeffs * self.sin_modes + cos_coeffs * self.cos_modes
        correction = -np.sum(correction, axis=0)
        self.dm.actuators[:] += correction
        self.dm_surface.append(self.dm.actuators)

        # return an image
        img = self.fwd(self.dm.render(wfe=True)) / self.ref_contrast

        # get mean in dark hole
        self.mean_in_dh.append(np.mean(img[self.dh==1]))

        return img
