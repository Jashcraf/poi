"""
# Optimizing APLC's with Artisan, Free-range, Hand-rolled, Algorithmic Differentiation

Greetings traveler, if you are here it's because you made some questionable choices in life that lead you to designing coronagraphs. Don't you know that the only way to achieve unaberrated imaging is by blocking all photons?

Regardless, herein lies an attempt at a tutorial to performing the APLC design methodology we wrote in Ashcraft et al. 2025. It leverages the algorithmic differentiation builtin to the `prysm` optical propagation package written by Brandon Dube, and an adaptation of a `vAPPOptimizer` written by Brandon. I also added a minor line of code to make L-BFGS-B happy using `cupy`.
"""

# The regular stuff
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import hwostyle
# hwostyle.use("light")
from tqdm import tqdm
from astropy.io import fits
from pathlib import Path
import time
import numpy as tnp
import sys

# The prysm stuff
from prysm.mathops import np, set_backend_to_cupy
from prysm.propagation import focus_fixed_sampling
from prysm.fttools import MatrixDFTExecutor

from poi.masks import ImgSamplingSpec, inner_core_mask, annular_mask, lyot_mask
from poi.aplc_design import APLCOptimizer, APLCWrapper, ThroughputOptimizer
import sys

AZMAGS = [60.0, 75.0, 90.0, 105.0, 120.0, 135.0, 150.0, 165.0, 180.0]
MULTIPLIERS = [10, 10, 10, 10, 10, 8, 10, 10, 7]
MULTIPLIERS = [10 for i in AZMAGS] 

# Not enough in the okabe cycle ;-;
okabe_colorblind8 = ['#E69F00', '#56B4E9', '#009E73',
                     '#F0E442', '#0072B2', '#D55E00', '#CC79A7',
                     '#000000', '#000000']

# Lets use plasma
cmap = plt.get_cmap('plasma')
n_lines = len(AZMAGS)
colors = [cmap(i / (n_lines - 1)) for i in range(n_lines)]

plt.figure()
plt.title("Field PSF Core Throughput")
# Construct coronagraph and build throughput plot
for color, AZMAG, MULTIPLIER in zip(colors, AZMAGS, MULTIPLIERS):
    IWA = 6
    # --- USER INPUT DESIGN PARAMS HERE
    USE_GPU = True # Use GPU for the optimization
    EPD = 24.4381  # milimeters
    EFL = EPD * 40 # milimeters
    WVL = 0.350 # microns
    IMG_NPIX = (256 + 128) // 1
    #IWA = 6
    OWA = 20
    AZMIN = -65 / 2 # Defines the angular extend of the dark zone
    AZMAX = 65 / 2
    BANDWIDTH = 10 # percent
    NWVLS = 5
    OVERSAMPLE = 8 # pix per lam/D
    pth_to_aperture = Path.home() / "poi/luvoir_b_pupil_512px.fits"
    LS_FRAC = 0.95 # Fraction of the pupil radius to use for the Lyot stop
    LS_OBSCURATION_RATIO = 0.00 # Ratio of the Lyot stop obscuration to the pupil radius
    core_size = 0.7 # radius in lam/D
    # 1e-11 produces good monochromatic designs

    # ---

    if USE_GPU:
        # np switches from numpy to cupy
        set_backend_to_cupy()

    tilt_lds = [12] # np.arange(0, OWA, 0.5)

    mdft = MatrixDFTExecutor()
    mdft.clear()

    # Set up the bandpass
    half_bw = BANDWIDTH / 2 / 100
    band = np.linspace(WVL * (1-half_bw), WVL * (1 + half_bw), NWVLS)

    # Set the FPM inner working angle and outer working angle to have margin before dark hole
    FPM_IWA = (1) * IWA
    FPM_OWA = (1) * OWA

    # Load the aperture
    aperture = np.round(np.array(fits.getdata(pth_to_aperture)))
    PUPIL_NPIX = aperture.shape[0]

    # Create the focal plane mask
    img_dx = WVL * (EFL / EPD) / OVERSAMPLE
    pupil_dx = EPD / PUPIL_NPIX

    psf = focus_fixed_sampling(wavefunction=aperture,
                               input_dx=pupil_dx,
                               prop_dist=EFL,
                               wavelength=WVL,
                               output_dx=img_dx,
                               output_samples=IMG_NPIX)

    lambd = EFL / EPD * WVL
    psf_scalar = np.abs(psf)**2
    iss = ImgSamplingSpec(IMG_NPIX, lambd / OVERSAMPLE, lambd)
    focal_plane_mask = annular_mask(iss, FPM_IWA, FPM_OWA, theta_min=AZMIN, theta_max=AZMAX)
    focal_plane_mask += np.fliplr(focal_plane_mask)
    dh = annular_mask(iss, IWA, OWA, theta_min=AZMIN, theta_max=AZMAX)
    dh = dh + np.fliplr(dh)
    ls_mask = lyot_mask(PUPIL_NPIX, pupil_dx=pupil_dx, frac=LS_FRAC, obscuration_ratio=LS_OBSCURATION_RATIO)

    # Load the computed mask
    mask_path = Path.home() / "Downloads/az_survey" \
                / f"LUVOIRB_512Npup_192Nimg_6IWA_20OWA_{AZMAG}AZ_0LSinner_0.95LSouter_{MULTIPLIER}throughput_weight.fits"
                    #LUVOIRB_512Npup_192Nimg_6IWA_20OWA_95.0AZ_0LSinner_0.95LSouter_13throughput_weight.fits
    newmask = np.array(fits.getdata(mask_path))


    def prop_coro(aplc, fpm, ls, wave=WVL, tilt=0, include_fpm=True):

        # Handle passing a float
        if isinstance(wave, float):
            wave = [wave]

        pupil_npix = PUPIL_NPIX


        before_fpm_intensity = 0
        before_ls_intensity = 0
        coro_img_onax_intensity = 0

        for wvl in wave:

            # get the tilt phase
            x = np.linspace(-0.5, 0.5, pupil_npix)
            tilt_phase = np.exp(1j * 2 * np.pi * x * tilt * (WVL / wvl))

            before_fpm = focus_fixed_sampling(
                        wavefunction= aplc * tilt_phase,
                        input_dx=pupil_dx,
                        prop_dist = EFL,
                        wavelength= wvl,
                        output_dx= img_dx,
                        output_samples=(IMG_NPIX, IMG_NPIX),
                        shift=(0, 0),
                        method='mdft')

            if include_fpm:
                before_fpm *= fpm

            before_ls = focus_fixed_sampling(
                        wavefunction=before_fpm,
                        input_dx=img_dx,
                        prop_dist = EFL,
                        wavelength= wvl,
                        output_dx= pupil_dx,
                        output_samples=(pupil_npix, pupil_npix),
                        shift=(0, 0),
                        method='mdft')

            coro_img_onax = focus_fixed_sampling(
                        wavefunction=before_ls * ls,
                        input_dx=pupil_dx,
                        prop_dist = EFL,
                        wavelength= wvl,
                        output_dx= img_dx,
                        output_samples=(IMG_NPIX, IMG_NPIX),
                        shift=(0, 0),
                        method='mdft')

            # accumulate intensities
            before_fpm_intensity += np.abs(before_fpm)**2
            before_ls_intensity += np.abs(before_ls)**2
            coro_img_onax_intensity += np.abs(coro_img_onax)**2

        return before_fpm_intensity, before_ls_intensity, coro_img_onax_intensity

    fx = np.linspace(-IMG_NPIX / (2 * OVERSAMPLE), IMG_NPIX / (2 * OVERSAMPLE), IMG_NPIX)
    fx, fy = np.meshgrid(fx, fx)

    # 14 mins uh oh
    from matplotlib.colors import LogNorm
    from tqdm import tqdm
    throughput = []
    throughput_07 = []

    # Get the reference intensity
    before_ref, ls, coro = prop_coro(aperture, focal_plane_mask, ls_mask, tilt=0, wave=band, include_fpm=False)
    fx = np.linspace(-IMG_NPIX / (2 * OVERSAMPLE), IMG_NPIX / (2 * OVERSAMPLE), IMG_NPIX)
    fx, fy = np.meshgrid(fx, fx)
    rx = np.sqrt(fx**2 + fy**2)
    mask = np.zeros_like(rx, dtype=int)
    mask[rx < core_size] = 1
    eta_0 = np.sum(before_ref[mask==1])
    limit = eta_0 / np.sum(NWVLS * aperture)

    for i, ld in tqdm(enumerate(tilt_lds)):

        before, ls, coro = prop_coro(newmask * aperture, focal_plane_mask, ls_mask, tilt=ld, wave=band)
        before_I = np.sum(before)
        coro_I = coro
        throughput.append(np.sum(coro_I) / np.sum(NWVLS * aperture))

        # get value in 0.7 L/D (recall 1/OS is pixelscale in L/D)
        fx = np.linspace(-IMG_NPIX / (2 * OVERSAMPLE), IMG_NPIX / (2 * OVERSAMPLE), IMG_NPIX)
        fx, fy = np.meshgrid(fx, fx)
        fx += ld
        rx = np.sqrt(fx**2 + fy**2)
        mask = np.zeros_like(rx, dtype=int)
        mask[rx < core_size] = 1

        # plt.figure()
        # plt.subplot(121)
        # plt.imshow((coro_I * mask).get(), norm=LogNorm())
        # plt.subplot(122)
        # plt.imshow((coro_I).get(), norm=LogNorm())
        # plt.show()

        value_in_aperture = np.sum(coro_I[mask==1])
        throughput_07 = value_in_aperture / eta_0
        plt.scatter(AZMAG, throughput_07.get(), marker="o", linestyle="None", color=colors[0], s=50)

#plt.plot(tilt_lds.get(), (np.ones_like(np.array(throughput_07)) * limit).get(), linestyle='dashed', color='black', label='Aperture Limit')
plt.xlabel('Azimuthal Extent, '+r'$\theta$')
plt.ylabel('Core Throughput, '+r'$r \leq 0.7 \lambda/D$')
plt.ylim(0, 1.)
plt.show()
