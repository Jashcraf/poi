from prysm.mathops import np
from tqdm import tqdm

def tikhonov_inverse(A, rcond=1e-3):
    """Compute a matrix pseudo-inverse using Tikhonov Regularization

    Parameters
    ----------
    A : ndarray
        2 dimensional ndarray to invert
    rcond : float, optional
        unsure, suspect it has to do with the spatial frequency supression,
        by default 1e-3

    Returns
    -------
    ndarray
        inverted matrix
    """
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    s_inv = s/(s**2 + (rcond * s.max())**2)
    return (Vt.T * s_inv).dot(U.T)

def beta_reg(J, beta=-2.5):
    """Compute a matrix pseudo-inverse using Beta Regularization

    Parameters
    ----------
    J : ndarray
        2 dimensional matrix to invert
    beta : float, optional
        power of spatial frequency supression, by default -2.5

    Returns
    -------
    ndarray
        beta-regularized inverted matrix
    """
    # J is the Jacobian
    JTJ = np.matmul(J.T, J)
    rho = np.diag(JTJ)
    alpha2 = rho.max()

    control_matrix = np.matmul( np.linalg.inv( JTJ + alpha2*10.0**(beta) * np.eye(JTJ.shape[0]) ), J.T)
    return control_matrix

class iEFC:

    def __init__(self, propagation, dm, modes, probes, dh, probe_amplitude=1., mode_amplitude=1., wavelength=1,
                 ref_contrast=1):
        """Instance of an implicit Electric Field Conjugation experiment,

        Substantial portions of this code were adapted from aefc_vortex, by Kian Milani
        https://github.com/kian1377/aefc-vortex/tree/main


        Parameters
        ----------
        propagation : callable
            function that takes a phasor as an argument and returns the electric field
            at the focal plane
        dm : prysm.x.DM
            prysm DM instance
        modes : list
            list of ndarrays containing the calibration modes
        probes : list
            list of ndarrays containing the measurement probes, must be
            length 2
        dh : boolean array
            boolean array containing the region of interest to dig a dark hole
        probe_amplitude : float, optional
            amplitude of the probes, radians, by default 1.
        mode_amplitude : float, optional
            amplitude of the modeas, radians, by default 1.
        wavelength : float, optional
            wavelength in microns, by default 1
        ref_contrast : float, optional
            maximum of the unocculted PSF
        """

        self.fwd = propagation
        self.dm = dm
        self.modes = modes
        self.probes = probes
        self.dh = dh
        self.probe_amplitude = probe_amplitude
        self.mode_amplitude = mode_amplitude
        self.kvec = 2 * np.pi / wavelength
        self.images = []
        self.mean_in_dh = []
        self.dm_surface = []
        self.ref_contrast = ref_contrast

    def measurement(self):
        """Take a difference of probe measurements to estimate the E-field

        Returns
        -------
        ndarray
            difference image used to estimate E-field
        """

        difference_images = []

        for i, probe in enumerate(self.probes):

            # apply the positive mode
            self.dm.actuators[:] += probe * self.probe_amplitude
            im_pos = self.fwd(self.dm.render(wfe=True)) / self.ref_contrast

            # remove surface
            self.dm.actuators[:] -= probe * self.probe_amplitude

            # apply the negative mode
            self.dm.actuators[:] += -1 * probe * self.probe_amplitude
            im_neg = self.fwd(self.dm.render(wfe=True)) / self.ref_contrast

            # remove the surface
            self.dm.actuators[:] -= -1 * probe * self.probe_amplitude

            diff_img = (im_pos - im_neg) / (2 * self.probe_amplitude)
            difference_images.append(diff_img)

        return np.asarray(difference_images)


    def calibrate(self):
        """Empirically calibrate EFC Response Matrix
        """

        response_matrix = []

        for i, calibration_mode in tqdm(enumerate(self.modes)):
            
            # Loading the response matrix
            response = 0

            for s in [-1, 1]:

                # apply to DM
                self.dm.actuators[:] += s * self.mode_amplitude * calibration_mode
                diff_ims = self.measurement()
                response += s * diff_ims / (2 * self.mode_amplitude)

                # remove from DM
                self.dm.actuators[:] -= s * self.mode_amplitude * calibration_mode

            if len(self.probes) == 2:
                response_matrix.append(np.concatenate([
                    response[0, self.dh==1],
                    response[1, self.dh==1]
                ]))

            else:
                raise NotImplementedError
            
        self.response_matrix = np.array(response_matrix).T

    def compute_control_matrix(self, beta=-2.5):
        """Invert the response matrix with beta regularization to get the
        control matrix

        Parameters
        ----------
        beta : float, optional
            see beta_reg docstring, by default -2.5
        """
        self.control_matrix = beta_reg(self.response_matrix, beta=beta)

    def step(self, loop_gain=1., leakage=0., update_probe_amplitude=None):
        """Advance the iEFC algorithm one iteration, computes the control matrix
        if it has not been computed yet.

        Parameters
        ----------
        loop_gain : float, optional
            gain of the updated DM actuator command, by default 1.
        leakage : float, optional
            how much the total command reduces as the iterations progress,
            by default 0.
        update_probe_amplitude : float, optional
            the amplitude to apply to the user-specified probes, by default None,
            which uses the last stored value of amplitude

        Returns
        -------
        ndarray
            image after the EFC correction is applied
        """

        if not hasattr(self, "control_matrix"):
            self.compute_control_matrix()

        if not hasattr(self, "total_command"):
            self.total_command = np.zeros(self.dm.Nact, dtype=np.float64)

        if update_probe_amplitude is not None:
            self.probe_amplitude = update_probe_amplitude

        if len(self.images) < 1:
            print("Taking starter image at position zero")
            
            # take a starter image
            img = self.fwd(self.dm.render(wfe=True)) / self.ref_contrast
            self.images.append(img)
            self.mean_in_dh.append(np.mean(img[self.dh==1]))

        diff_ims = self.measurement()
        measurement_vector = diff_ims[:, self.dh==1].ravel()
        modal_matrix = np.asarray(self.modes).reshape(len(self.modes), -1)

        modal_coeff = -self.control_matrix.dot(measurement_vector)
        del_command = modal_matrix.T.dot(modal_coeff).reshape(self.dm.Nact)
        self.total_command = (1 - leakage) * self.total_command + loop_gain * del_command

        # remove the mean command
        # self.total_command -= np.mean(self.total_command)

        self.dm.actuators[:] = self.total_command

        # take an image
        img = self.fwd(self.dm.render(wfe=True)) / self.ref_contrast
        self.images.append(img)
        self.mean_in_dh.append(np.mean(img[self.dh==1]))
        self.dm_surface.append(self.dm.render())

        return img