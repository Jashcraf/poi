from prysm.mathops import np
from tqdm import tqdm

def tikhonov_inverse(A, rcond=1e-3):
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    s_inv = s/(s**2 + (rcond * s.max())**2)
    return (Vt.T * s_inv).dot(U.T)

def beta_reg(J, beta=-2):
    # J is the Jacobian
    JTJ = np.matmul(J.T, J)
    rho = np.diag(JTJ)
    alpha2 = rho.max()

    control_matrix = np.matmul( np.linalg.inv( JTJ + alpha2*10.0**(beta) * np.eye(JTJ.shape[0]) ), J.T)
    return control_matrix

class iEFC:

    def __init__(self, propagation, dm, modes, probes, dh, probe_amplitude=1., mode_amplitude=1., wavelength=1):
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

    def measurement(self):

        difference_images = []

        for i, probe in enumerate(self.probes):

            # apply the positive mode
            self.dm.actuators[:] += probe * self.probe_amplitude
            im_pos = np.abs(self.fwd(np.exp(1j * self.dm.render(wfe=True))))**2

            # remove surface
            self.dm.actuators[:] -= probe * self.probe_amplitude

            # apply the negative mode
            self.dm.actuators[:] += -1 * probe * self.probe_amplitude
            im_neg = np.abs(self.fwd(np.exp(1j * self.dm.render(wfe=True))))**2

            # remove the surface
            self.dm.actuators[:] -= -1 * probe * self.probe_amplitude

            diff_img = (im_pos - im_neg) / (2 * self.probe_amplitude)
            difference_images.append(diff_img)

        return np.asarray(difference_images)


    def calibrate(self):

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

    def compute_control_matrix(self):
        self.control_matrix = beta_reg(self.response_matrix)

    def step(self, loop_gain=1., leakage=0., update_probe_amplitude=None):

        if not hasattr(self, "control_matrix"):
            self.compute_control_matrix()

        if not hasattr(self, "total_command"):
            self.total_command = np.zeros(self.dm.Nact)

        if update_probe_amplitude is not None:
            self.probe_amplitude = update_probe_amplitude

        if len(self.images) < 1:
            print("Taking starter image at position zero")
            
            # take a starter image
            img = np.abs(self.fwd(np.exp(1j * self.dm.render(wfe=True))))**2
            self.images.append(img)

        diff_ims = self.measurement()
        measurement_vector = diff_ims[:, self.dh==1].ravel()
        modal_matrix = np.asarray(self.modes).reshape(len(self.modes), -1)

        modal_coeff = -self.control_matrix.dot(measurement_vector)
        del_command = modal_matrix.T.dot(modal_coeff).reshape(self.dm.Nact)
        self.total_command = (1 - leakage) * self.total_command + loop_gain * del_command

        self.dm.actuators[:] += self.total_command

        # take an image
        img = np.abs(self.fwd(np.exp(1j * self.dm.render(wfe=True))))**2
        self.images.append(img)
        self.mean_in_dh.append(np.mean(img[self.dh==1]))

        return img