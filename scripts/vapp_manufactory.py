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
import ipdb

# The prysm stuff
from prysm.mathops import np, set_backend_to_cupy
from prysm.propagation import focus_fixed_sampling
from prysm.fttools import MatrixDFTExecutor
from prysm._richdata import Slices

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

from poi.aplc_design import ImgSamplingSpec, inner_core_mask, annular_mask, lyot_mask, knife_edge_mask, circular_mask
from poi.aplc_design import PAPCOptimizer, APLCWrapper, ThroughputOptimizer, CoreThroughputOptimizer 


# --- USER INPUT DESIGN PARAMS HERE
USE_GPU = True # Use GPU for the optimization
EPD = 24.4381  # milimeters
EFL = EPD * 40 # milimeters
WVL = 1.0 # microns
IMG_NPIX = 256 + 128
IWA = 5
OWA = 10
AZMIN = -89 # Defines the angular extend of the dark zone
AZMAX = 89
BANDWIDTH = 40 # percent
NWVLS = 5
OVERSAMPLE = 8 # pix per lam/D
pth_to_aperture = Path.home() / "poi/hex_pupil_amplitude_6510mm_1024pix.fits"
LS_FRAC = 0.85 # Fraction of the pupil radius to use for the Lyot stop
LS_OBSCURATION_RATIO = 0.0 # Ratio of the Lyot stop obscuration to the pupil radius
MAX_ITERS = 10000
core_size = 0.7 # radius in lam/D
# 1e-11 produces good monochromatic designs
THROUGHPUT_RELATIVE_WEIGHT =  1e-10 # 1e-15 # relative weight of the throughput optimization
# ---

if USE_GPU:
    # np switches from numpy to cupy
    set_backend_to_cupy()

tilt_lds = np.arange(0, OWA, 0.25)

mdft = MatrixDFTExecutor()
mdft.clear()

# Set up the bandpass
half_bw = BANDWIDTH / 2 / 100
band = np.linspace(WVL * (1-half_bw), WVL * (1 + half_bw), NWVLS)
print(band)

# Set the FPM inner working angle and outer working angle to have margin before dark hole
FPM_IWA = IWA#(1 + half_bw) * IWA
FPM_OWA = OWA#(1 - half_bw) * OWA

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
focal_plane_mask = annular_mask(iss, FPM_IWA, FPM_OWA)
focal_plane_mask *= knife_edge_mask(iss, IWA)
dh = annular_mask(iss, IWA, OWA)
dh *= knife_edge_mask(iss, IWA)
#dh = np.fliplr(dh)
ls_mask = lyot_mask(PUPIL_NPIX, pupil_dx=pupil_dx, frac=LS_FRAC, obscuration_ratio=LS_OBSCURATION_RATIO)

plt.figure(figsize=[15,5])
plt.style.use('default')
plt.subplot(131)
plt.title('Dark Hole')
if np.__name__ == "cupy":
    plt.imshow(dh.get(), cmap='gray')
else:
    plt.imshow(dh, cmap='gray')
plt.subplot(132)
plt.title('Focal Plane Mask')
if np.__name__ == "cupy":
    plt.imshow(focal_plane_mask.get(), cmap='gray')
else:
    plt.imshow(focal_plane_mask, cmap='gray')
plt.subplot(133)
plt.title('Lyot stop')
if np.__name__ == "cupy":
    plt.imshow(ls_mask.get(), cmap='gray')
else:
    plt.imshow(ls_mask, cmap='gray')

# Break hermetian symmetry with a little bit of random noise
noisy = np.random.random(aperture.shape) * aperture / 1000
optlist = []
core_window = circular_mask(iss, 0.35)

for wave in band:
    aplc = PAPCOptimizer(amp = aperture,
                        amp_dx=pupil_dx,
                        efl=EFL,
                        wvl=wave,
                        basis=None,
                        dark_hole=dh,
                        dh_target=1e-2, # allows for specific contrast targeting, 0 just means "make it dark pls"
                        dh_dx=img_dx)
    aplc.set_optimization_method(zonal=True)
    optlist.append(aplc)


    throughput = CoreThroughputOptimizer(amp=aperture,
                                     amp_dx=pupil_dx,
                                     efl=EFL,
                                     wvl=wave,
                                     basis=None,
                                     window=(1-core_window), 
                                     dh_dx=img_dx,
                                     fpm=focal_plane_mask,
                                     ls=ls_mask,
                                     relative_weight=THROUGHPUT_RELATIVE_WEIGHT)

    throughput.set_optimization_method(zonal=True)
    optlist.append(throughput)


# optimization wrapper that sums the gradients and objective functions
opt_contrast_throughput = APLCWrapper(optlist=optlist)

# starting guess is a filled aperture
if np.__name__ == "cupy":
    x0 = tnp.random.random(aplc.amp.get().shape)[aplc.amp_select.get()]
else:
    x0 = tnp.random.random(aplc.amp.shape)[aplc.amp_select]
x0 /= 100

# Dry-run to debug
opt_contrast_throughput.fg(x0)

# initialize the optimizer with box constraints
opt = F77LBFGSB(opt_contrast_throughput.fg, x0, memory=5,
                upper_bounds=tnp.ones(x0.shape),
                lower_bounds=-tnp.ones(x0.shape))
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

newmask = aplc.amp.astype(tnp.complex128)
newmask[aplc.amp_select] = tnp.exp(1j * 2 * np.pi / WVL * opt.x)

plt.style.use("default")
plt.figure(figsize=[15,5])
plt.subplot(131)
plt.title('Aperture')
if np.__name__ == "cupy":
    plt.imshow(aperture.get(),cmap='gray')
else:
    plt.imshow(aperture, cmap='gray')
plt.colorbar()
plt.subplot(132)
plt.title('Phase Apodizer')
if np.__name__ == "cupy":
    plt.imshow(tnp.angle(newmask.get()),cmap='RdBu_r')
else:
    plt.imshow(tnp.angle(newmask), cmap='RdBu_r')
plt.colorbar(label="microns")
plt.subplot(133)
plt.title('Lyot Stop Field')
if np.__name__ == "cupy":
    plt.imshow(ls_mask.get(),cmap='inferno')
else:
    plt.imshow(ls_mask,cmap='inferno')
plt.colorbar()

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
        tilt_phase = np.exp(-1j * 2 * np.pi * x * tilt * WVL / wvl)

        before_fpm = focus_fixed_sampling(
                    wavefunction= aplc * tilt_phase,
                    input_dx=pupil_dx,
                    prop_dist = EFL,
                    wavelength= wvl,
                    output_dx= img_dx,
                    output_samples=(IMG_NPIX, IMG_NPIX),
                    shift=(0, 0),
                    method='mdft')

        # accumulate intensities
        coro_img_onax_intensity += np.abs(before_fpm)**2

    return coro_img_onax_intensity

fx = np.linspace(-IMG_NPIX / (2 * OVERSAMPLE), IMG_NPIX / (2 * OVERSAMPLE), IMG_NPIX)
fx, fy = np.meshgrid(fx, fx)

# 14 mins uh oh
from matplotlib.colors import LogNorm
from tqdm import tqdm
throughput = []
throughput_07 = []

# Get the contrast normalization
before = prop_coro(newmask, focal_plane_mask, ls_mask, tilt=0, include_fpm=False, wave=band)
contrast_norm = before.max()

contrast_onax = before / contrast_norm

for i, ld in tqdm(enumerate(tilt_lds)):

    before  = prop_coro(newmask, focal_plane_mask, ls_mask, tilt=ld, wave=band)
    before_I = np.sum(newmask)
    coro_I = before 

    if ld == 10:
        plt.figure()
        plt.title(f"Tilt = {ld}")
        plt.imshow(coro_I.get(), cmap="inferno", norm=LogNorm(vmax=1e4))
        plt.colorbar()

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
plt.style.use("bmh")
colors = plt.rcParams['axes.prop_cycle'].by_key()['color'][1:]

def radial_profile(data, center=[int(IMG_NPIX/2),int(IMG_NPIX/2)]):
    y, x = tnp.indices((data.shape))
    r = tnp.sqrt((x - center[0])**2 + (y - center[1])**2)
    r = r.astype(int)

    tbin = tnp.bincount(r.ravel(), data.ravel())
    nr = tnp.bincount(r.ravel())
    radialprofile = tbin / nr
    return radialprofile

plt.figure(figsize=[12,4])
plt.subplot(121)
if np.__name__ == "cupy":
    plt.plot(tilt_lds.get(), np.array(throughput).get(), linestyle='dashed', color=colors[0])
    plt.plot(tilt_lds.get(), np.array(throughput_07).get(), linestyle='solid', color=colors[0])
    plt.plot(tilt_lds.get(), -np.array(throughput).get(), linestyle='solid', color='black', label=r'$r = 0.7\lambda / D$')
    plt.plot(tilt_lds.get(), -np.array(throughput).get(), linestyle='dashed', color='black', label=r'$r = \infty$')
else:
    plt.plot(tilt_lds, np.array(throughput), linestyle='dashed', color=colors[0])
    plt.plot(tilt_lds, np.array(throughput_07), linestyle='solid', color=colors[0])
    plt.plot(tilt_lds, -np.array(throughput), linestyle='solid', color='black', label=r'$r = 0.7\lambda / D$')
# plt.vlines(3.5,-1,1, color=colors[0], alpha=0.5)
# plt.vlines(2.5,-1,1, color=colors[1], alpha=0.5)
plt.xlabel('Angular Separation, '+r'$\lambda / D$')
# plt.text(3, 0.15, 'APLC-3.5 IWA', rotation=90, color=colors[0], fontweight='bold')
# plt.text(2, 0.15, 'APLC-2.5 IWA', rotation=90, color=colors[1], fontweight='bold')
plt.ylabel('Throughput')
plt.legend(loc='lower right')
plt.ylim(0,1)
plt.xlim(0, OWA)



# get using prysm slices
x = np.linspace(-IMG_NPIX / 2, IMG_NPIX / 2, IMG_NPIX)
y = np.copy(x)
dh_nanmask = np.copy(dh)
dh[dh < 1] = np.nan
contrast_slice = Slices(contrast_onax * dh_nanmask, x, y)
points, radial_profile = contrast_slice.azavg
#radial_profile = radial_profile((contrast_onax * dh).get())
x_axis = tnp.ones_like(radial_profile.get()) # just get array size
dx_ld = 1/OVERSAMPLE/2 # pixelscale in lambda/D
x_ticks = [dx_ld*i for i in range(len(x_axis))]
x_ticks = tnp.array(x_ticks)

plt.subplot(122)
plt.plot(x_ticks, radial_profile.get(), color=colors[0], label='PAPLC')
# plt.vlines(3.5,0,1, color=colors[0], alpha=0.5, linestyle='solid')
# plt.vlines(2.5,0,1, color=colors[1], alpha=0.5, linestyle='solid')
plt.xlabel('Angular Separation, '+r'$\lambda / D$')
plt.xlim(0, OWA)
plt.ylim(1e-12, 1)
plt.yscale('log')
plt.ylabel('Normalized Intensity')
plt.legend()
# plt.savefig('coronagraph_throughput_and_contrast.pdf')

plt.style.use("default")
plt.figure()
plt.imshow(contrast_onax.get(), cmap="inferno", norm=LogNorm(vmin=1e-12, vmax=1))
plt.colorbar(label="Normalized Intensity")

plt.show()
