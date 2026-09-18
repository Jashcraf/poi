"""
# Optimizing APLC's with Artisan, Free-range, Hand-rolled, Algorithmic Differentiation

Greetings traveler, if you are here it's because you made some questionable choices in life that lead you to designing coronagraphs. Don't you know that the only way to achieve unaberrated imaging is by blocking all photons?

Regardless, herein lies an attempt at a tutorial to performing the APLC design methodology we wrote in Ashcraft et al. 2025. It leverages the algorithmic differentiation builtin to the `prysm` optical propagation package written by Brandon Dube, and an adaptation of a `vAPPOptimizer` written by Brandon. I also added a minor line of code to make L-BFGS-B happy using `cupy`.
"""

# The regular stuff
import hwostyle
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import pandas as pd 

hwostyle.use("light")
import sys
import time
from pathlib import Path

import numpy as tnp
from astropy.io import fits
from prysm.fttools import MatrixDFTExecutor

# The prysm stuff
from prysm.mathops import np, set_backend_to_cupy
from prysm.propagation import focus_fixed_sampling
from tqdm import tqdm

from poi.aplc_design import APLCOptimizer, APLCWrapper, ThroughputOptimizer
from poi.masks import ImgSamplingSpec, annular_mask, inner_core_mask, lyot_mask

LS_RADII = np.arange(0.7, 0.95, 0.01)  # [0.80, 0.85, 0.90, 0.95]

MULTIPLIERS = np.full_like(LS_RADII, 10.0)
# MULTIPLIERS[-2:] = 9.
FAILED_SOLUTION = np.ones_like(LS_RADII)

# Not enough in the okabe cycle ;-;
okabe_colorblind8 = [
    "#E69F00",
    "#56B4E9",
    "#009E73",
    "#F0E442",
    "#0072B2",
    "#D55E00",
    "#CC79A7",
    "#000000",
    "#000000",
]
ls_types = ["Circle", "Edges", "Segments"]
markers = ["o", "s", "^"]
# Lets use plasma
cmap = plt.get_cmap("plasma")
n_lines = len(LS_RADII)
colors = okabe_colorblind8  # [cmap(i / (n_lines - 1)) for i in range(n_lines)]

# Load up AJ's SPLCs
df = pd.read_csv('ampl_splc_data.csv')
od_ls_splc = df['OD_LS']
thpt_circl_splc = df['thput_ee_rel_circular']
thpt_hex_splc = df['thput_ee_rel_hex']
print(od_ls_splc)
print(thpt_circl_splc)
print(thpt_hex_splc)

plt.figure()
plt.title(r"$10 \lambda_0 / D_C$" + " PSF Core Throughput")
for color, ls_type, marker in zip(colors, ls_types, markers):
    # Construct coronagraph and build throughput plot
    for LS_FRAC, MULTIPLIER, FAILED in zip(LS_RADII, MULTIPLIERS, FAILED_SOLUTION):
        if LS_FRAC == 0.95 and ls_type == "Segments":
            continue

        if LS_FRAC == 0.70 and ls_type == "Edges":
            continue
        IWA = 6
        # --- USER INPUT DESIGN PARAMS HERE
        USE_GPU = False  # Use GPU for the optimization
        EPD = 24.4381  # milimeters
        EFL = EPD * 40  # milimeters
        WVL = 0.350  # microns
        IMG_NPIX = (256 + 128) // 1
        # IWA = 6
        OWA = 20
        AZMIN = -65 / 2  # Defines the angular extend of the dark zone
        AZMAX = 65 / 2

        NWVLS = 5
        OVERSAMPLE = 8  # pix per lam/D
        pth_to_aperture = Path.home() / "poi/luvoir_b_pupil_512px.fits"
        LS_OBSCURATION_RATIO = (
            0.00  # Ratio of the Lyot stop obscuration to the pupil radius
        )
        BANDWIDTH = 10
        core_size = 0.7  # radius in lam/D
        # 1e-11 produces good monochromatic designs

        # ---

        if USE_GPU:
            # np switches from numpy to cupy
            set_backend_to_cupy()

        tilt_lds = [10]  # np.arange(0, OWA, 0.5)

        mdft = MatrixDFTExecutor()
        mdft.clear()

        # Set up the bandpass
        half_bw = BANDWIDTH / 2 / 100
        band = np.linspace(WVL * (1 - half_bw), WVL * (1 + half_bw), NWVLS)

        # Set the FPM inner working angle and outer working angle to have margin before dark hole
        FPM_IWA = (1) * IWA
        FPM_OWA = (1) * OWA

        # Load the aperture
        aperture = np.round(np.array(fits.getdata(pth_to_aperture)))
        PUPIL_NPIX = aperture.shape[0]

        # Create the focal plane mask
        img_dx = WVL * (EFL / EPD) / OVERSAMPLE
        pupil_dx = EPD / PUPIL_NPIX

        psf = focus_fixed_sampling(
            wavefunction=aperture,
            input_dx=pupil_dx,
            prop_dist=EFL,
            wavelength=WVL,
            output_dx=img_dx,
            output_samples=IMG_NPIX,
        )

        lambd = EFL / EPD * WVL
        psf_scalar = np.abs(psf) ** 2
        iss = ImgSamplingSpec(IMG_NPIX, lambd / OVERSAMPLE, lambd)
        focal_plane_mask = annular_mask(
            iss, FPM_IWA, FPM_OWA, theta_min=AZMIN, theta_max=AZMAX
        )
        focal_plane_mask += np.fliplr(focal_plane_mask)
        dh = annular_mask(iss, IWA, OWA, theta_min=AZMIN, theta_max=AZMAX)
        dh = dh + np.fliplr(dh)

        # Load the computed mask
        mask_path = (
            Path.home()
            / f"Downloads/Data/10.0/LS_{ls_type}_{int(LS_FRAC * 100)}_01_20260601"
        )
        apod_path = mask_path / f"apodizer_0.35cenwvl_5wvls_10%.fits"

        ls_path = mask_path / f"lyot_stop.fits"

        # LUVOIRB_512Npup_192Nimg_6IWA_20OWA_95.0AZ_0LSinner_0.95LSouter_13throughput_weight.fits
        newmask = np.array(fits.getdata(apod_path))
        ls_mask = np.array(fits.getdata(ls_path))

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
                    wavefunction=aplc * tilt_phase,
                    input_dx=pupil_dx,
                    prop_dist=EFL,
                    wavelength=wvl,
                    output_dx=img_dx,
                    output_samples=(IMG_NPIX, IMG_NPIX),
                    shift=(0, 0),
                    method="mdft",
                )

                if include_fpm:
                    before_fpm *= fpm

                before_ls = focus_fixed_sampling(
                    wavefunction=before_fpm,
                    input_dx=img_dx,
                    prop_dist=EFL,
                    wavelength=wvl,
                    output_dx=pupil_dx,
                    output_samples=(pupil_npix, pupil_npix),
                    shift=(0, 0),
                    method="mdft",
                )

                coro_img_onax = focus_fixed_sampling(
                    wavefunction=before_ls * ls,
                    input_dx=pupil_dx,
                    prop_dist=EFL,
                    wavelength=wvl,
                    output_dx=img_dx,
                    output_samples=(IMG_NPIX, IMG_NPIX),
                    shift=(0, 0),
                    method="mdft",
                )

                # accumulate intensities
                before_fpm_intensity += np.abs(before_fpm) ** 2
                before_ls_intensity += np.abs(before_ls) ** 2
                coro_img_onax_intensity += np.abs(coro_img_onax) ** 2

            return before_fpm_intensity, before_ls_intensity, coro_img_onax_intensity

        fx = np.linspace(
            -IMG_NPIX / (2 * OVERSAMPLE), IMG_NPIX / (2 * OVERSAMPLE), IMG_NPIX
        )
        fx, fy = np.meshgrid(fx, fx)

        # 14 mins uh oh
        from matplotlib.colors import LogNorm
        from tqdm import tqdm

        throughput = []
        throughput_07 = []

        # Get the reference intensity
        before_ref, ls, coro = prop_coro(
            aperture, focal_plane_mask, ls_mask, tilt=0, wave=band, include_fpm=False
        )
        fx = np.linspace(
            -IMG_NPIX / (2 * OVERSAMPLE), IMG_NPIX / (2 * OVERSAMPLE), IMG_NPIX
        )
        fx, fy = np.meshgrid(fx, fx)
        rx = np.sqrt(fx**2 + fy**2)
        mask = np.zeros_like(rx, dtype=int)
        mask[rx < core_size] = 1
        eta_0 = np.sum(before_ref[mask == 1])
        limit = eta_0 / np.sum(NWVLS * aperture)

        for i, ld in tqdm(enumerate(tilt_lds)):
            before, ls, coro = prop_coro(
                newmask * aperture, focal_plane_mask, ls_mask, tilt=ld, wave=band
            )
            before_I = np.sum(before)
            coro_I = coro
            throughput.append(np.sum(coro_I) / np.sum(NWVLS * aperture))

            # get value in 0.7 L/D (recall 1/OS is pixelscale in L/D)
            fx = np.linspace(
                -IMG_NPIX / (2 * OVERSAMPLE), IMG_NPIX / (2 * OVERSAMPLE), IMG_NPIX
            )
            fx, fy = np.meshgrid(fx, fx)
            fx += ld
            rx = np.sqrt(fx**2 + fy**2)
            mask = np.zeros_like(rx, dtype=int)
            mask[rx < core_size] = 1

            value_in_aperture = np.sum(coro_I[mask == 1])
            throughput_07 = value_in_aperture / eta_0
            if hasattr(throughput_07, "get"):
                throughput_07 = throughput_07.get()
            plt.scatter(
                LS_FRAC,
                throughput_07,
                marker=marker,
                linestyle="None",
                c=color,
                s=50,
            )


# Plot AJ designs
plt.scatter(od_ls_splc, thpt_circl_splc, marker=markers[0], label="Gurobi Circular")
plt.scatter(od_ls_splc / 1.07, thpt_hex_splc, marker=markers[1], label="Gurobi Degraded Edges")

# plt.plot(tilt_lds.get(), (np.ones_like(np.array(throughput_07)) * limit).get(), linestyle='dashed', color='black', label='Aperture Limit')
plt.xlabel("Equiv. LS Radius, " + r"$ D_{LS} / D_{EP}$")
plt.ylabel("E.E. Relative Throughput, " + r"$r \leq 0.7 \lambda/D$")
plt.ylim(0.10, 0.65)
plt.xlim(0.69, 0.95)
plt.scatter(-10, -10, marker=markers[0], c=colors[0], s=50, label="Circular")
plt.scatter(-10, -10, marker=markers[1], c=colors[1], s=50, label="Degraded Edges")
plt.scatter(-10, -10, marker=markers[2], c=colors[2], s=50, label="Degraded Segments")
plt.legend()
plt.savefig("throughput_lyotstops.png")
plt.show()
