# The regular stuff
from math import e
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.gridspec as gridspec
from tqdm import tqdm
from astropy.io import fits
from pathlib import Path
import time
import numpy as tnp

# The prysm stuff
from poi.masks import ImgSamplingSpec
from prysm.mathops import np, set_backend_to_cupy
from prysm.propagation import focus_fixed_sampling, unfocus_fixed_sampling
from prysm.fttools import MatrixDFTExecutor

from poi.masks import ImgSamplingSpec, inner_core_mask, annular_mask

# --- USER INPUT DESIGN PARAMS HERE
USE_GPU = True # Use GPU for the optimization
EPD = 24.4381  # milimeters
EFL = EPD * 40 # milimeters
WVL = 0.350 # microns
IMG_NPIX = 256
IWA = 6
OWA = 20
AZMIN = -65 / 2 # Defines the angular extend of the dark zone
AZMAX = 65 / 2
BANDWIDTH = .10 # percent
NWVLS = 1
OVERSAMPLE = 4 # pix per lam/D
core_size = 0.7 # radius in lam/D

# Load up the data
parent = Path.home() / "Downloads/APLC_20.5_2026-02-18_14-09-36"
pth_to_aperture =  parent / "aperture.fits"
pth_to_apodizer = parent / "apodizer_0.35cenwvl_3wvls_10%.fits"
pth_to_fpm = parent / "fpm_6IWA_20OWA_4OS_-32.5min_32.5max.fits"
pth_to_lyot = parent / "lyot_stop.fits"

aperture = fits.getdata(pth_to_aperture)
apodizer = fits.getdata(pth_to_apodizer)
fpm = fits.getdata(pth_to_fpm)
lyot = fits.getdata(pth_to_lyot)

# Try to draw a disk FPM
lambd = EFL / EPD * WVL
# iss = ImgSamplingSpec(fpm.shape[0], lambd/OVERSAMPLE, lambd)
# fpm = inner_core_mask(iss, 6)
# fpm = 1.- fpm.astype(float)
# fpm = annular_mask(iss, 6, 20)

masks = [fpm, lyot]
labels = ["FPM", "Lyot Stop"]

# sampling parameters
PUPIL_NPIX = aperture.shape[0]
img_dx = WVL * (EFL / EPD) / OVERSAMPLE
pupil_dx = EPD / PUPIL_NPIX

# Set up the bandpass
tilt_lds = np.arange(0, OWA, 0.05)
half_bw = BANDWIDTH / 2 / 100
band = np.linspace(WVL * (1-half_bw), WVL * (1 + half_bw), NWVLS)

# Construct the field intensities
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
        tilt_phase = np.exp(-1j * 2 * np.pi * x * tilt * (WVL / wvl))

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
            after_fpm = before_fpm * fpm
        else:
            after_fpm = before_fpm

        before_ls = unfocus_fixed_sampling(
                    wavefunction=after_fpm,
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

# Generate the fields
before_fpm, before_ls, coro_img = prop_coro(aplc=apodizer,
                                            fpm=fpm,
                                            ls=lyot,
                                            wave=band,
                                            tilt=0,
                                            include_fpm=True)

# Set up the matplotlib gridspec
fig = plt.figure(figsize=(16, 6))
main_grid = gridspec.GridSpec(2, 4, figure=fig, wspace=0.3, hspace=0.4)

top_axes = []
bottom_axes = []

for group in range(2):  # 4 groups of (1 top centered over 2 bottom)
    col = group * 2

    # Top image: spans 2 columns
    ax_top = fig.add_subplot(main_grid[0, col:col+2])
    top_axes.append(ax_top)

    # Bottom-left image
    ax_bl = fig.add_subplot(main_grid[1, col])
    bottom_axes.append(ax_bl)

    # Bottom-right image
    ax_br = fig.add_subplot(main_grid[1, col+1])
    bottom_axes.append(ax_br)

# Load up the masks
for i, ax in enumerate(top_axes):
    ax.imshow(masks[i], cmap='gray')
    ax.set_title(f'{labels[i]}')
    ax.axis('off')

fields = [
    before_fpm,
    before_fpm * fpm,

    before_ls,
    before_ls * lyot,
]

for i, ax in enumerate(bottom_axes):
    if i == 0 or i == 1:
        ax.imshow(fields[i], cmap='jet', norm=LogNorm(vmin=None, vmax=None))
    else:
        ax.imshow(fields[i], cmap='jet')
    ax.axis('off')

plt.suptitle('Design Plotter', fontsize=14)
plt.show()
