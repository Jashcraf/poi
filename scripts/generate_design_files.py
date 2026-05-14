# The regular stuff
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as tnp
from astropy.io import fits
from matplotlib.colors import LogNorm
from prysm.fttools import MatrixDFTExecutor

# The prysm stuff
from prysm.mathops import np, set_backend_to_cupy
from prysm.propagation import focus_fixed_sampling

# Available optimizers
from prysm.x.optym import (
    F77LBFGSB,  # The one that works with Box constraints
    AdaGrad,
    Adam,
    AdaMomentum,
    GradientDescent,
    RAdam,
    RMSProp,
    # These are untested, but need to include something to bound the solution
    Yogi,
)
from tqdm import tqdm

from poi.aplc_design import (
    APLCOptimizer,
    APLCWrapper,
    ThroughputOptimizer,
)
from poi.masks import ImgSamplingSpec, annular_mask, inner_core_mask, lyot_mask

# Handle incoming args
if len(sys.argv) > 1:
    THROUGHPUT_LOG_WEIGHT = float(sys.argv[1])
    # BANDWIDTH = float(sys.argv[2])
    LS_FRAC = float(sys.argv[2])
else:
    THROUGHPUT_LOG_WEIGHT = 20
    # BANDWIDTH = 10
    LS_FRAC = 0.95

# --- USER INPUT DESIGN PARAMS HERE
USE_GPU = True  # Use GPU for the optimization
EPD = 24.4381  # milimeters
EFL = EPD * 40  # milimeters
WVL = 0.350  # microns
IMG_NPIX = (256 + 128) // 2
IWA = 6
OWA = 20
AZMAG = 65
AZMIN = -AZMAG / 2  # Defines the angular extend of the dark zone
AZMAX = AZMAG / 2
OVERSAMPLE = 4  # pix per lam/D
BANDWIDTH = 10
pth_to_aperture = Path.home() / "poi/luvoir_b_pupil_512px.fits"
# LS_FRAC = 0.95 # Fraction of the pupil radius to use for the Lyot stop
LS_OBSCURATION_RATIO = 0  # Ratio of the Lyot stop obscuration to the pupil radius
MAX_ITERS = 100_000
core_size = 0.7  # radius in lam/D
THROUGHPUT_RELATIVE_WEIGHT = 10 ** (-1 * THROUGHPUT_LOG_WEIGHT)

SAVE_DIR = Path.home() / "poi/Data"

if USE_GPU:
    # np switches from numpy to cupy
    set_backend_to_cupy()

# Configure save directory, makes directory if it doesn't exist
from datetime import datetime

now = datetime.now()
filename = f"APLC_{THROUGHPUT_LOG_WEIGHT}_{now.strftime('%Y-%m-%d_%H-%M-%S')}"
SAVE_DIR = SAVE_DIR / filename
os.makedirs(SAVE_DIR, exist_ok=True)

tilt_lds = np.arange(0, OWA, 0.05)

mdft = MatrixDFTExecutor()
mdft.clear()

# Set up the bandpass
# Set up the bandpass
half_bw = BANDWIDTH / 2 / 100

# compute NWVLS so that there's at least one wavelength per 4%, also add a
# catch to make the minimum number of wavelengths 5. This means that it will
# only start adding wavelengths after 20%. Using np.ceil here to always use
# more wavelengths than we strictly need.
NWVLS = int(np.ceil(BANDWIDTH / 4))
if NWVLS < 5:
    NWVLS = 5

band = np.linspace(WVL * (1 - half_bw), WVL * (1 + half_bw), NWVLS)

# Set the FPM inner working angle and outer working angle to have margin before dark hole
FPM_IWA = IWA
FPM_OWA = OWA

# Load the aperture
aperture = np.array(fits.getdata(pth_to_aperture))
PUPIL_NPIX = aperture.shape[0]

# Save the aperture before any data manipulation
if hasattr(aperture, "get"):
    ap_to_save = aperture.get()
else:
    ap_to_save = np.copy(aperture)

hdu_aper = fits.PrimaryHDU(ap_to_save.astype(tnp.float64))
hdu_aper.writeto(SAVE_DIR / f"aperture.fits")

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

if np.__name__ == "cupy":
    psf = (psf_scalar / psf_scalar.max()).get()
else:
    psf = psf_scalar / psf_scalar.max()

# Draw coronagraph masks
iss = ImgSamplingSpec(IMG_NPIX, lambd / OVERSAMPLE, lambd)
focal_plane_mask = annular_mask(iss, FPM_IWA, FPM_OWA, theta_min=AZMIN, theta_max=AZMAX)
focal_plane_mask += np.fliplr(focal_plane_mask)
dh = annular_mask(iss, IWA, OWA, theta_min=AZMIN, theta_max=AZMAX)
dh = dh + np.fliplr(dh)
ls_mask = lyot_mask(
    PUPIL_NPIX, pupil_dx=pupil_dx, frac=LS_FRAC, obscuration_ratio=LS_OBSCURATION_RATIO
)


optlist = []

# Use optimizers that don't assume symmetry
for wave in band:
    aplc = APLCOptimizer(
        amp=aperture,
        amp_dx=pupil_dx,
        efl=EFL,
        wvl=wave,
        basis=None,
        dark_hole=dh,
        dh_target=0,  # allows for specific contrast targeting, 0 just means "make it dark pls"
        dh_dx=img_dx,
        fpm=focal_plane_mask,
        ls=ls_mask,
        weight=1e10,
    )
    aplc.set_optimization_method(zonal=True)
    optlist.append(aplc)

throughput = ThroughputOptimizer(
    amp=aperture,
    wvl=WVL,
    basis=None,
    ls=np.ones_like(ls_mask),
    relative_weight=THROUGHPUT_RELATIVE_WEIGHT * NWVLS,
)

throughput.set_optimization_method(zonal=True)
optlist.append(throughput)

# Set up CoreThroughput cost function
core_mask = inner_core_mask(iss, core_size)

# optimization wrapper that sums the gradients and objective functions
opt_contrast_throughput = APLCWrapper(optlist=optlist)

# starting guess is a filled aperture
# If point_symmetric=True, this filters out the aperture
if np.__name__ == "cupy":
    x0 = tnp.ones(aplc.amp.get().shape, dtype=float)[aplc.amp_select.get()]
else:
    x0 = tnp.ones(aplc.amp.shape, dtype=float)[aplc.amp_select]


# Dry-run to debug
_, _ = opt_contrast_throughput.fg(x0)


# initialize the optimizer with box constraints
opt = F77LBFGSB(
    opt_contrast_throughput.fg,
    x0,
    memory=10,
    upper_bounds=tnp.ones(x0.shape),
    lower_bounds=tnp.zeros(x0.shape),
)
opt.iprint = 0

# some timing
t1 = time.perf_counter()

# This is in a try-except block because the optimizer will
# sometimes raise a StopIteration exception when it is done

# This is in a try-except block because the optimizer will
# sometimes raise a StopIteration exception when it is done
try:
    for _ in tqdm(range(MAX_ITERS)):
        opt.step()
except StopIteration:
    pass
print(f"Time to Optimizer for {MAX_ITERS}")
print(time.perf_counter() - t1)

newmask = aplc.amp
newmask[aplc.amp_select] = opt.x

plt.style.use("bmh")
fig = plt.figure(figsize=[20, 10])
gs = fig.add_gridspec(2, 4)

# Set up axes
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[0, 2])
ax4 = fig.add_subplot(gs[0, 3])

ax5 = fig.add_subplot(gs[1, 0:2])
ax6 = fig.add_subplot(gs[1, 2:4])

ax1.set_title("Pupil Apodizer")
if np.__name__ == "cupy":
    ax1.imshow(newmask.get(), cmap="gray")
else:
    ax1.imshow(newmask, cmap="gray")

# Clear ticks
ax1.set_xticks([])
ax1.set_xticklabels([])
ax1.set_yticks([])
ax1.set_yticklabels([])

ax2.set_title("Focal Plane Mask")
if np.__name__ == "cupy":
    ax2.imshow(focal_plane_mask.get(), cmap="gray")
else:
    ax2.imshow(focal_plane_mask, cmap="gray")

# Clear ticks
ax2.set_xticks([])
ax2.set_xticklabels([])
ax2.set_yticks([])
ax2.set_yticklabels([])

okabe_colorblind8 = [
    "#000000",
    "#E69F00",
    "#56B4E9",
    "#009E73",
    "#F0E442",
    "#0072B2",
    "#D55E00",
    "#CC79A7",
]


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


fx = np.linspace(-IMG_NPIX / (2 * OVERSAMPLE), IMG_NPIX / (2 * OVERSAMPLE), IMG_NPIX)
fx, fy = np.meshgrid(fx, fx)

# 14 mins uh oh
from matplotlib.colors import LogNorm
from tqdm import tqdm

throughput = []

throughput_07 = []

# Get the contrast normalization
before, _, coro = prop_coro(
    newmask, focal_plane_mask, ls_mask, tilt=0, include_fpm=False, wave=band
)
contrast_norm = coro.max()

before, lyot_field, coro = prop_coro(
    newmask, focal_plane_mask, ls_mask, tilt=0, include_fpm=True, wave=band
)
contrast_onax = coro / contrast_norm

# Lyot Stop plot
ax3.set_title("Lyot Field")
if np.__name__ == "cupy":
    ax3.imshow(lyot_field.get(), cmap="inferno", norm=LogNorm())
else:
    ax3.imshow(lyot_field, cmap="inferno", norm=LogNorm())

# Clear ticks
ax3.set_xticks([])
ax3.set_xticklabels([])
ax3.set_yticks([])
ax3.set_yticklabels([])

# Focal Plane plot
ax4.set_title("Starlight")
if np.__name__ == "cupy":
    ax4.imshow(contrast_onax.get(), cmap="inferno", norm=LogNorm(vmin=1e-11, vmax=1e-5))
else:
    ax4.imshow(contrast_onax, cmap="inferno", norm=LogNorm())

# Clear ticks
ax4.set_xticks([])
ax4.set_xticklabels([])
ax4.set_yticks([])
ax4.set_yticklabels([])


for i, ld in tqdm(enumerate(tilt_lds)):
    before, ls, coro = prop_coro(newmask, focal_plane_mask, ls_mask, tilt=ld, wave=band)
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
    throughput_07.append(value_in_aperture / np.sum(NWVLS * aperture))

# get the bmh colors
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"][1:]


ax5.set_title("Throughput")
if np.__name__ == "cupy":
    ax5.plot(
        tilt_lds.get(), np.array(throughput).get(), linestyle="dashed", color=colors[0]
    )
    ax5.plot(
        tilt_lds.get(),
        np.array(throughput_07).get(),
        linestyle="solid",
        color=colors[0],
    )
    ax5.plot(
        tilt_lds.get(),
        -np.array(throughput).get(),
        linestyle="solid",
        color="black",
        label=r"$r = 0.7\lambda / D$",
    )
    ax5.plot(
        tilt_lds.get(),
        -np.array(throughput).get(),
        linestyle="dashed",
        color="black",
        label=r"$r = \infty$",
    )
else:
    ax5.plot(tilt_lds, np.array(throughput), linestyle="dashed", color=colors[0])
    ax5.plot(tilt_lds, np.array(throughput_07), linestyle="solid", color=colors[0])
    ax5.plot(
        tilt_lds,
        -np.array(throughput),
        linestyle="solid",
        color="black",
        label=r"$r = 0.7\lambda / D$",
    )
ax5.set_xlabel("Angular Separation, " + r"$\lambda / D$")
ax5.set_ylabel("Throughput")
ax5.legend(loc="lower right")
ax5.set_ylim(0, 1)
ax5.set_xlim(0, OWA)

from poi.processing import azimuthal_average

# get radial
masked_contrast = contrast_onax * dh
radial_profile, bins = azimuthal_average(
    masked_contrast.get(), angle_range=[AZMIN, AZMAX]
)
x_axis = tnp.ones_like(radial_profile)  # just get array size
dx_ld = 1 / OVERSAMPLE  # pixelscale in lambda/D
x_ticks = [dx_ld * i for i in bins]
x_ticks = tnp.array(x_ticks)

LS_OUTER = LS_FRAC
LS_INNER = LS_OBSCURATION_RATIO

ax6.set_title("Contrast Curve")
ax6.plot(x_ticks, radial_profile, color=colors[0], label="APLC")
ax6.set_xlabel("Angular Separation, " + r"$\lambda / D$")
ax6.set_xlim(0, OWA)
ax6.set_ylim(1e-12, 1e-5)
ax6.set_yscale("log")
ax6.set_ylabel("Normalized Intensity")
ax6.legend()
plt.savefig(
    SAVE_DIR
    / f"LUVOIR_B_{PUPIL_NPIX}Npup_{IMG_NPIX}Nimg_{IWA}IWA_{OWA}OWA_{AZMAG}AZ_{LS_INNER}LSinner_{LS_OUTER}LSouter_{THROUGHPUT_LOG_WEIGHT}throughput_weight_{BANDWIDTH}bw_{NWVLS}wls.pdf"
)

# Clean up and save
mdft.clear()

# Save coronagraph masks
if hasattr(newmask, "get"):
    newmask = newmask.get()

if hasattr(focal_plane_mask, "get"):
    focal_plane_mask = focal_plane_mask.get()

if hasattr(ls_mask, "get"):
    ls_mask = ls_mask.get()

# Construct fits headers
hdu_apod = fits.PrimaryHDU(newmask.astype(tnp.float64))
hdu_fpm = fits.PrimaryHDU(focal_plane_mask.astype(tnp.float64))
hdu_lyot = fits.PrimaryHDU(ls_mask.astype(tnp.float64))

# Add sampling parameters to apodizer
#################00000000############
hdu_apod.header["NPIX"] = newmask.shape[0]
hdu_apod.header["DX"] = (EPD, "[mm]")
hdu_apod.header["F/#"] = 40
hdu_apod.header["EFL"] = (EFL, "[mm]")
hdu_apod.header["CEN WVL"] = (WVL, "[microns]")
hdu_apod.header["IWA [lam/D]"] = (IWA, "[lam/D]")
hdu_apod.header["OWA [lam/D]"] = (OWA, "[lam/D]")
hdu_apod.header["AZMIN"] = (AZMIN, "[deg]")
hdu_apod.header["AZMAX"] = (AZMAX, "[deg]")
hdu_apod.header["BW"] = (BANDWIDTH, "[%]")
hdu_apod.header["NWVLS"] = NWVLS
hdu_apod.header["OVERSAMPLE"] = (OVERSAMPLE, "[pix/lam/D]")
hdu_apod.header["LS FRAC RADIUS"] = LS_FRAC
hdu_apod.header["LS OBST RADIUS"] = LS_OBSCURATION_RATIO

# Add focal-plane specific parameters to FPM
hdu_fpm.header["FPM IWA"] = (FPM_IWA, "[lam/D]")
hdu_fpm.header["FPM OWA"] = (FPM_OWA, "[lam/D]")
hdu_fpm.header["DX"] = (1 / OVERSAMPLE, "[lam/D/pix]")
hdu_fpm.header["NPIX"] = focal_plane_mask.shape[0]


hdu_apod.writeto(SAVE_DIR / f"apodizer_{WVL}cenwvl_{NWVLS}wvls_{BANDWIDTH}%.fits")
hdu_lyot.writeto(SAVE_DIR / f"lyot_stop.fits")
hdu_fpm.writeto(
    SAVE_DIR / f"fpm_{IWA}IWA_{OWA}OWA_{OVERSAMPLE}OS_{AZMIN}min_{AZMAX}max.fits"
)
