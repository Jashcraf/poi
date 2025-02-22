from prysm.mathops import np
from prysm.coordinates import make_xy_grid, cart_to_polar
from .modes import fourier_modes_sequence

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

    def __init__(self, propagation, dx_img, epd, efl, wvl, dm, IWA, OWA, ref_contrast=1):

        self.fwd = propagation
        self.dm = dm
        self.dx_img = dx_img
        self.epd = epd
        self.efl = efl
        self.wvl = wvl
        self.image_dx_lamD = self.dx_img * 1e-6 / (self.efl * 1e-3) * (self.epd * 1e-3) / (self.wvl * 1e-6)
        self.IWA = IWA
        self.OWA = OWA
        self.kvec = 2 * np.pi / wvl
        self.images = []
        self.mean_in_dh = []
        self.dm_surface = []
        self.ref_contrast = ref_contrast
        self.nsteps = 4
        self.nact = self.dm.Nact

        # construct a dark hole
        self.Nimg = self.fwd().shape[0]
        x, y = make_xy_grid(self.Nimg, dx=self.image_dx_lamD)
        r, t = cart_to_polar(x, y)

        self.dh = np.zeros([self.Nimg, self.Nimg])
        self.dh[r < self.OWA] = 1.
        self.dh[r < self.IWA] = 0.

        # Build up the fourier mode basis
        # - recall: real modes are cosines within DH,
        #           imag modes are sines within DH

        self.real_modes = fourier_modes_sequence(self.nact, which="cos")
        self.real_probe = np.sum(self.real_modes, axis=0)
        self.real_probe /= np.max(self.real_probe)
        self.imag_modes = fourier_modes_sequence(self.nact, which="sin")
        self.imag_probe = np.sum(self.imag_modes, axis=0)
        self.imag_probe /= np.max(self.imag_probe)

        pass

    def step(self):
        
        # Starting image acquisition

        # Four probe steps

        # Reconstruct DM commands

        # Apply correction
        
        pass


