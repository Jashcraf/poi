"""
# Optimizing APLC's with Artisan, Free-range, Hand-rolled, Algorithmic Differentiation

Greetings traveler, if you are here it's because you made some questionable choices in life that lead you to designing coronagraphs. Don't you know that the only way to achieve unaberrated imaging is by blocking all photons?

Regardless, herein lies an attempt at a tutorial to performing the APLC design methodology we wrote in Ashcraft et al. 2025. It leverages the algorithmic differentiation builtin to the `prysm` optical propagation package written by Brandon Dube, and an adaptation of a `vAPPOptimizer` written by Brandon. I also added a minor line of code to make L-BFGS-B happy using `cupy`.
"""

# The regular stuff
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from tqdm import tqdm
from astropy.io import fits
from pathlib import Path
import time
import numpy as tnp
import sys
from tqdm import tqdm

# The prysm stuff
from prysm.mathops import np, set_backend_to_cupy
from prysm.propagation import focus_fixed_sampling
from prysm.fttools import MatrixDFTExecutor

from poi.aplc_design import ImgSamplingSpec, inner_core_mask, annular_mask, lyot_mask
from poi.aplc_design import APLCOptimizer, APLCWrapper, ThroughputOptimizer
import sys

lyot_stop_inner = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
lyot_stop_outer = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
throughput_array = np.zeros([len(lyot_stop_inner), len(lyot_stop_outer)])
okabe_colorblind8 = ['#E69F00', '#56B4E9', '#009E73',
                     '#F0E442', '#0072B2', '#D55E00', '#CC79A7','#000000']


# Construct coronagraph and build throughput plot
for i, LS_OBSCURATION_RATIO in enumerate(lyot_stop_inner):
    for j, LS_FRAC in enumerate(lyot_stop_outer):

        # --- USER INPUT DESIGN PARAMS HERE
        USE_GPU = True # Use GPU for the optimization
        EPD = 24.4381  # milimeters
        EFL = EPD * 40 # milimeters
        WVL = 0.350 # microns
        IMG_NPIX = (256 + 128) // 1
        IWA = 6
        OWA = 20
        AZMIN = -65 / 2 # Defines the angular extend of the dark zone
        AZMAX = 65 / 2
        BANDWIDTH = 10 # percent
        NWVLS = 5
        OVERSAMPLE = 8 # pix per lam/D
        pth_to_aperture = Path.home() / "poi/luvoir_b_pupil_512px.fits"
        #LS_FRAC = 0.9 # Fraction of the pupil radius to use for the Lyot stop
        #LS_OBSCURATION_RATIO = 0.00 # Ratio of the Lyot stop obscuration to the pupil radius
        MAX_ITERS = 100_000
        core_size = 0.7 # radius in lam/D
        # 1e-11 produces good monochromatic designs

        THROUGHPUT_LOG_WEIGHT = 13
        THROUGHPUT_RELATIVE_WEIGHT = 10 ** (-1 * THROUGHPUT_LOG_WEIGHT)
        # ---

        if USE_GPU:
            # np switches from numpy to cupy
            set_backend_to_cupy()

        tilt_ld = 10 # places PSF at 10 lambda / D to evaluate throughput

        mdft = MatrixDFTExecutor()
        mdft.clear()

        # Set up the bandpass
        half_bw = BANDWIDTH / 2 / 100
        band = np.linspace(WVL * (1-half_bw), WVL * (1 + half_bw), NWVLS)

        # Set the FPM inner working angle and outer working angle to have margin before dark hole
        FPM_IWA = (1 + half_bw) * IWA
        FPM_OWA = (1 - half_bw) * OWA

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
        mask_path = Path.home() / "Data/LS_Survey_11112025-selected" \
                    / f"LUVOIRB_512Npup_192Nimg_6IWA_20OWA_65AZ_{LS_OBSCURATION_RATIO}LSinner_{LS_FRAC}LSouter_13throughput_weight.fits"


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



        before, ls, coro = prop_coro(newmask, focal_plane_mask, ls_mask, tilt=tilt_ld, wave=band)
        before_I = np.sum(before)
        coro_I = coro

        # get value in 0.7 L/D (recall 1/OS is pixelscale in L/D)
        fx = np.linspace(-IMG_NPIX / (2 * OVERSAMPLE), IMG_NPIX / (2 * OVERSAMPLE), IMG_NPIX)
        fx, fy = np.meshgrid(fx, fx)
        fx += tilt_ld
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
        throughput_array[i, j] = value_in_aperture / (NWVLS * np.sum(aperture))
    

# Find the maximum index
idx = tnp.where(throughput_array == throughput_array.max())
yind, xind = idx


plt.figure()
plt.title(f"Core Throughput at {tilt_ld}"+r"$\lambda_0 / D_c$")
plt.plot(xind, yind, linestyle="None", marker="x", color="k", markersize=10, label="Max Throughput")
plt.imshow(throughput_array, cmap="GnBu", origin="lower", vmin=0.15, vmax=0.4)
plt.plot(xind, yind, linestyle="None", marker="x", color="w", markersize=10)
plt.legend()
ax = plt.gca()
ax.set_xticks(np.arange(len(lyot_stop_outer)).get())
ax.set_yticks(np.arange(len(lyot_stop_inner)).get())
ax.set_xticklabels(lyot_stop_outer)
ax.set_yticklabels(lyot_stop_inner)
plt.xlabel('Lyot Stop Outer Diameter '+r"$D / D_c$")
plt.ylabel('Lyot Stop Inner Diameter '+r"$D / D_c$")
plt.colorbar()
plt.show()

