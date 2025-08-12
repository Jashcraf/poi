from prysm.interferogram import (
    abc_psd,
    render_synthetic_surface
)
from prysm.mathops import np
from prysm.coordinates import make_xy_grid, cart_to_polar
from prysm.propagation import focus_fixed_sampling, unfocus_fixed_sampling
from prysm.geometry import circle
from prysm.x.dm import DM

# pound the poi
from poi.influence_funcs import gaussian_influence_function
from poi.modes import fourier_modes_sequence
from poi.speckle_nulling import find_image_center, compute_pixel_scale_lamd
from poi.speckle_nulling import SpeckleAreaNulling

# plotting needs
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle, Wedge

DISPLAY_INTERMEDIATES = False
a = 5e4
b = 1/1000
c = 3

WVL = 1 # micron
EFL = 100 # mm
Dpup = 10 # mm
Npup = 512
dx_img = 1/2 # microns

nact = 22
act_pitch = 3
samples_per_act = 21

# dark hole
IWA = 4
OWA = 10
CORO_IWA = 3
ANGLE_RANGE = [-np.pi/2, np.pi/2]

if __name__ == '__main__':

    # set up the Deformable mirror
    sampling_pitch = act_pitch / samples_per_act

    x, y = make_xy_grid(Npup, dx=sampling_pitch)
    r, th = cart_to_polar(x, y)
    influence_func = gaussian_influence_function(r, act_pitch)
    influence_func /= np.max(influence_func)

    Nout = x.shape[0]
    dm = DM(influence_func, Nout, nact, samples_per_act, rot=(0,0,0), shift=(0,0))
    dm.act_pitch = act_pitch

    # set up the physical optics model
    x, y = make_xy_grid(Npup, diameter=2)
    r, t = cart_to_polar(x, y)
    A = circle(1, r)
    LS = circle(0.9, r)

    # create pupil function and propagate
    wvfnt = np.copy(A)

    # Sample field
    focal = focus_fixed_sampling(wvfnt,
                                 input_dx=Dpup / wvfnt.shape[0], # mm pupil
                                 prop_dist=EFL, # mm focal length
                                 wavelength=WVL, # microns
                                 output_dx=dx_img, # microns
                                 output_samples=wvfnt.shape[0])

    # create a dark hole
    um_to_LD = 1e-6 / (EFL * 1e-3) * (Dpup * 1e-3) / (WVL * 1e-6)
    u, v = make_xy_grid(focal.shape, dx=dx_img * um_to_LD)
    rho, psi = cart_to_polar(u, v)

    dark_hole = np.zeros_like(u)
    dark_hole[rho < OWA] = 1
    dark_hole[rho < IWA] = 0
    dark_hole[psi < ANGLE_RANGE[0]] = 0
    dark_hole[psi > ANGLE_RANGE[1]] = 0

    FPM = np.ones_like(u)
    FPM[ rho < CORO_IWA] = 0

    focal_intensity = np.abs(focal)**2

    # plotting the coronagraph
    if DISPLAY_INTERMEDIATES == True:
        plt.figure(figsize=[15,3])
        plt.subplot(141)
        plt.imshow(A, cmap='gray')
        plt.xticks([],[])
        plt.yticks([],[])
        plt.title('Telescope Aperture')
        plt.subplot(142)
        plt.imshow(FPM, cmap='gray')
        plt.title('Focal Plane Mask')
        plt.xticks([],[])
        plt.yticks([],[])
        plt.subplot(143)
        plt.imshow(LS, cmap='gray')
        plt.title('Lyot Stop')
        plt.xticks([],[])
        plt.yticks([],[])
        plt.subplot(144)
        plt.imshow(dark_hole, cmap='gray')
        plt.title('Dark Hole')
        plt.xticks([],[])
        plt.yticks([],[])
        # plt.show()

    # Set up the PSD
    nu = np.arange(1, 1000, 1)
    psd = abc_psd(nu, a=a, b=b, c=c)

    # Just picked ABC values until it looked about right
    Npup = Nout
    x, y, z = render_synthetic_surface(10, Npup, a=a, b=b, c=c)

    if DISPLAY_INTERMEDIATES == True:

        plt.figure()
        plt.title('Synthetic WFE')
        plt.imshow(z, cmap='RdBu_r')
        plt.colorbar(label='microns WFE')
        # plt.show()

        plt.figure()
        plt.plot(nu, psd)
        plt.xscale('log')
        plt.yscale('log')
        plt.ylabel('PSD')
        plt.xlabel('Spatial Frequency')
        # plt.show()

    # Define a propagation funciton using the coronagraph
    def snap_image(pupil_wfe=np.zeros_like(A), include_fpm=True, wvls=np.array([0.95*WVL, WVL, 1.05*WVL])):
        """take intensity image given a pupil wfe in microns

        Parameters
        ----------
        pupil_wfe : ndarray, optional
            wavefront error in microns, by default np.ones_like(A)
        include_fpm : bool, optional
            include the focal plane mask, by default True
        wvls : ndarray, optional
            wavelengths in microns, by default np.array([0.95*WVL, WVL, 1.05*WVL])
        """

        # create nominal wavefront
        x, y = make_xy_grid(z.shape, dx=10/Npup)
        cosine_wfe = 0 #np.cos(3*x + y) / 50
        psf = 0

        for wvl in wvls:

            wfe = np.exp(1j * 2 * np.pi / wvl * (pupil_wfe + cosine_wfe + z))
            wvfnt_to_prop = wfe * wvfnt

            # propagate to focus
            focal = focus_fixed_sampling(wvfnt_to_prop,
                                        input_dx=Dpup / wvfnt.shape[0], # mm pupil
                                        prop_dist=EFL, # mm focal length
                                        wavelength=wvl, # microns
                                        output_dx=dx_img, # microns
                                        output_samples=wvfnt.shape[0])

            if include_fpm:
                focal *= FPM

            lyot = unfocus_fixed_sampling(focal,
                                        input_dx=dx_img, # um
                                        prop_dist=EFL, # mm focal length
                                        wavelength=wvl, # microns
                                        output_dx=Dpup / wvfnt.shape[0], # mm
                                        output_samples=wvfnt.shape[0])

            lyot *= LS

            coro = focus_fixed_sampling(lyot,
                                        input_dx=Dpup / wvfnt.shape[0], # mm pupil
                                        prop_dist=EFL, # mm focal length
                                        wavelength=wvl, # microns
                                        output_dx=dx_img, # microns
                                        output_samples=wvfnt.shape[0])
            psf += (np.abs(coro)**2) / len(wvls)

        return psf

    # compute image from satelite spots
    nominal_image = snap_image()
    waffle = np.ones_like(dm.actuators) * 1e-1
    for i in range(dm.actuators.shape[0]):
        for j in range(dm.actuators.shape[1]):

            if i%2 != 0 or j%2 != 0:
                waffle[i,j] = 0

    dm.actuators[:] += waffle


    waffle_image = snap_image(dm.render(wfe=True))
    dh_region = Wedge([256, 256], OWA/(dx_img * um_to_LD),
                      theta1=-90, theta2=90, width=(OWA-IWA)/(dx_img * um_to_LD),
                      edgecolor="w", facecolor="None")
    dm.actuators[:] -= waffle
    dm.actuators[:] = 0

    row_cen, col_cen = find_image_center(waffle_image)
    dx_lamD_empirical = 1/compute_pixel_scale_lamd(waffle_image, waffle_frequency=nact//2)
    print(dx_lamD_empirical)
    print((dx_img * um_to_LD))

    # Init the algorithm
    san = SpeckleAreaNulling(propagation=snap_image,
                            dx_img=dx_img,
                            epd=Dpup,
                            efl=EFL,
                            wvl=WVL,
                            dm=dm,
                            IWA=IWA,
                            OWA=OWA,
                            ref_contrast=1,
                            edge=1,
                            angular_range=[-90, 90])
    reg = 0.99
    for i in range(50):
        img = san.step(regularization=reg**i)
    plt.figure()
    plt.imshow(img, cmap="inferno", norm=LogNorm())
    plt.colorbar()
    plt.show()

    plt.show()
