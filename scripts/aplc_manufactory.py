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

# The prysm stuff
from prysm.mathops import np, set_backend_to_cupy
from prysm.propagation import focus_fixed_sampling
from prysm.fttools import MatrixDFTExecutor


# Available optimizers
from prysm.x.optym import (
    F77LBFGSB, # The one that works with Box constraints

    # These are untested, but need to include something to bound the solution
    Yogi,
    Adam,
    GradientDescent,
    AdaGrad,
    RMSProp,
    RAdam,
    AdaMomentum
)

from poi.aplc_design import ImgSamplingSpec, inner_core_mask, annular_mask, lyot_mask
from poi.aplc_design import APLCOptimizer, APLCWrapper, ThroughputOptimizer


# --- USER INPUT DESIGN PARAMS HERE
USE_GPU = True # Use GPU for the optimization
EPD = 24.4381  # milimeters
EFL = EPD * 40 # milimeters
WVL = 0.350 # microns
IMG_NPIX = 256 + 128
IWA = 6
OWA = 20
AZMIN = -65 / 2 # Defines the angular extend of the dark zone
AZMAX = 65 / 2
BANDWIDTH = 10 # percent
NWVLS = 3
OVERSAMPLE = 8 # pix per lam/D
pth_to_aperture = Path.home() / "poi/hex_pupil_amplitude_6510mm_1024pix.fits"
pth_to_aperture = Path.home() / "poi/luvoir_b_pupil_512px.fits"
LS_FRAC = 0.9 # Fraction of the pupil radius to use for the Lyot stop
LS_OBSCURATION_RATIO = 0.00 # Ratio of the Lyot stop obscuration to the pupil radius
MAX_ITERS = 100
core_size = 0.7 # radius in lam/D
# 1e-11 produces good monochromatic designs
THROUGHPUT_RELATIVE_WEIGHT =  1e-12 # 1e-15 # relative weight of the throughput optimization
# ---

if USE_GPU:
    # np switches from numpy to cupy
    set_backend_to_cupy()

tilt_lds = np.arange(0, OWA, 0.05)

mdft = MatrixDFTExecutor()
mdft.clear()

# Set up the bandpass
half_bw = BANDWIDTH / 2 / 100
band = np.linspace(WVL * (1-half_bw), WVL * (1 + half_bw), NWVLS)
print(band)

# Set the FPM inner working angle and outer working angle to have margin before dark hole
FPM_IWA = (1 + half_bw) * IWA
FPM_OWA = (1 - half_bw) * OWA

# Load the aperture
aperture = np.array(fits.getdata(pth_to_aperture))
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

plt.figure()
if np.__name__ == "cupy":
    psf = (psf_scalar / psf_scalar.max()).get()
else:
    psf = psf_scalar / psf_scalar.max()
plt.imshow(psf, norm=LogNorm())
plt.colorbar(label="Normalized Intensity")

iss = ImgSamplingSpec(IMG_NPIX, lambd / OVERSAMPLE, lambd)
focal_plane_mask = annular_mask(iss, FPM_IWA, FPM_OWA, theta_min=AZMIN, theta_max=AZMAX)
focal_plane_mask += np.fliplr(focal_plane_mask)
dh = annular_mask(iss, IWA, OWA, theta_min=AZMIN, theta_max=AZMAX)
dh = dh + np.fliplr(dh)
ls_mask = lyot_mask(PUPIL_NPIX, pupil_dx=pupil_dx, frac=LS_FRAC, obscuration_ratio=LS_OBSCURATION_RATIO)


# Break hermetian symmetry with a little bit of random noise
noisy = np.random.random(aperture.shape) * aperture / 1000
optlist = []
for wave in band:
    aplc = APLCOptimizer(amp = aperture - noisy,
                        amp_dx=pupil_dx,
                        efl=EFL,
                        wvl=wave,
                        basis=None,
                        dark_hole=dh,
                        dh_target=0, # allows for specific contrast targeting, 0 just means "make it dark pls"
                        dh_dx=img_dx,
                        fpm=focal_plane_mask,
                        ls=ls_mask)
    aplc.set_optimization_method(zonal=True)
    optlist.append(aplc)

throughput = ThroughputOptimizer(amp=aperture-noisy,
                                 wvl=WVL,
                                 basis=None,
                                 ls=ls_mask,
                                 relative_weight=THROUGHPUT_RELATIVE_WEIGHT * NWVLS)

throughput.set_optimization_method(zonal=True)
optlist.append(throughput)

# optimization wrapper that sums the gradients and objective functions
opt_contrast_throughput = APLCWrapper(optlist=optlist)

# starting guess is a filled aperture
if np.__name__ == "cupy":
    x0 = tnp.ones(aplc.amp.get().shape, dtype=float)[aplc.amp_select.get()]
else:
    x0 = tnp.ones(aplc.amp.shape, dtype=float)[aplc.amp_select]

# Dry-run to debug
opt_contrast_throughput.fg(x0)

# initialize the optimizer with box constraints
opt = F77LBFGSB(opt_contrast_throughput.fg, x0,
                memory=5, upper_bounds=tnp.ones(x0.shape),
                lower_bounds=tnp.zeros(x0.shape))
opt.iprint = 0

# some timing
t1 = time.perf_counter()

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
fig = plt.figure(figsize=[20,10])
gs = fig.add_gridspec(2, 3)

# Set up axes
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])
ax3 = fig.add_subplot(gs[0,2])

ax4 = fig.add_subplot(gs[1, 0])
ax5 = fig.add_subplot(gs[1, 1:3])

ax1.set_title('Pupil Apodizer')
if np.__name__ == "cupy":
    ax1.imshow(newmask.get(),cmap='gray')
else:
    ax1.imshow(newmask, cmap='gray')

# Clear ticks
ax1.set_xticks([])
ax1.set_xticklabels([])
ax1.set_yticks([])
ax1.set_yticklabels([])

ax2.set_title('Focal Plane Mask')
if np.__name__ == "cupy":
    ax2.imshow(focal_plane_mask.get(), cmap='gray')
else:
    ax2.imshow(focal_plane_mask, cmap='gray')

# Clear ticks
ax2.set_xticks([])
ax2.set_xticklabels([])
ax2.set_yticks([])
ax2.set_yticklabels([])

okabe_colorblind8 = ['#000000', '#E69F00', '#56B4E9', '#009E73',
                     '#F0E442', '#0072B2', '#D55E00', '#CC79A7']

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

# Get the contrast normalization
before, _, coro = prop_coro(newmask, focal_plane_mask, ls_mask, tilt=0, include_fpm=False, wave=band)
contrast_norm = before.max()

before, lyot_field, coro = prop_coro(newmask, focal_plane_mask, ls_mask, tilt=0, include_fpm=True, wave=band)
contrast_onax = coro / contrast_norm

# Lyot Stop plot
ax3.set_title('Lyot Field')
if np.__name__ == "cupy":
    ax3.imshow(lyot_field.get(), cmap='inferno', norm=LogNorm())
else:
    ax3.imshow(lyot_field, cmap='inferno', norm=LogNorm())

# Clear ticks
ax3.set_xticks([])
ax3.set_xticklabels([])
ax3.set_yticks([])
ax3.set_yticklabels([])


for i, ld in tqdm(enumerate(tilt_lds)):

    before, ls, coro = prop_coro(newmask, focal_plane_mask, ls_mask, tilt=ld, wave=band)
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

    value_in_aperture = np.sum(coro_I[mask==1])
    throughput_07.append(value_in_aperture / np.sum(NWVLS * aperture))

# get the bmh colors
colors = plt.rcParams['axes.prop_cycle'].by_key()['color'][1:]


ax4.set_title('Throughput')
if np.__name__ == "cupy":
    ax4.plot(tilt_lds.get(), np.array(throughput).get(), linestyle='dashed', color=colors[0])
    ax4.plot(tilt_lds.get(), np.array(throughput_07).get(), linestyle='solid', color=colors[0])
    ax4.plot(tilt_lds.get(), -np.array(throughput).get(), linestyle='solid', color='black', label=r'$r = 0.7\lambda / D$')
    ax4.plot(tilt_lds.get(), -np.array(throughput).get(), linestyle='dashed', color='black', label=r'$r = \infty$')
else:
    ax4.plot(tilt_lds, np.array(throughput), linestyle='dashed', color=colors[0])
    ax4.plot(tilt_lds, np.array(throughput_07), linestyle='solid', color=colors[0])
    ax4.plot(tilt_lds, -np.array(throughput), linestyle='solid', color='black', label=r'$r = 0.7\lambda / D$')
# plt.vlines(3.5,-1,1, color=colors[0], alpha=0.5)
# plt.vlines(2.5,-1,1, color=colors[1], alpha=0.5)
ax4.set_xlabel('Angular Separation, '+r'$\lambda / D$')
# plt.text(3, 0.15, 'APLC-3.5 IWA', rotation=90, color=colors[0], fontweight='bold')
# plt.text(2, 0.15, 'APLC-2.5 IWA', rotation=90, color=colors[1], fontweight='bold')
ax4.set_ylabel('Throughput')
ax4.legend(loc='lower right')
ax4.set_ylim(0,1)
ax4.set_xlim(0, OWA)

from poi.processing import azimuthal_average

# get radial
masked_contrast = contrast_onax * dh
radial_profile, bins = azimuthal_average(masked_contrast.get(), angle_range=[AZMIN, AZMAX])
x_axis = tnp.ones_like(radial_profile) # just get array size
dx_ld = 1/OVERSAMPLE # pixelscale in lambda/D
x_ticks = [dx_ld*i for i in bins]
x_ticks = tnp.array(x_ticks)

ax5.set_title("Contrast Curve")
ax5.plot(x_ticks, radial_profile, color=colors[0], label='APLC')
# plt.vlines(3.5,0,1, color=colors[0], alpha=0.5, linestyle='solid')
# plt.vlines(2.5,0,1, color=colors[1], alpha=0.5, linestyle='solid')
ax5.set_xlabel('Angular Separation, '+r'$\lambda / D$')
ax5.set_xlim(0, OWA)
ax5.set_ylim(1e-12, 1e-5)
ax5.set_yscale('log')
ax5.set_ylabel('Normalized Intensity')
ax5.legend()
plt.savefig('coronagraph_throughput_and_contrast.pdf')
plt.show()
