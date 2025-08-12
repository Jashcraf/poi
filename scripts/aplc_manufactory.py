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

# Available optimizers
from prysm.x.optym import (
    F77LBFGSB, # 
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

# np switches from numpy to cupy
set_backend_to_cupy()

# --- USER INPUT DESIGN PARAMS HERE
EPD = 24.4381  # milimeters
EFL = EPD * 40 # milimeters
WVL = 0.656 # microns
IMG_NPIX = 256
IWA = 5
OWA = 12
AZMIN = -89
AZMAX = 89
BANDWIDTH = 10 # percent
NWVLS = 5
OVERSAMPLE = 8
pth_to_aperture = Path.home() / "poi/hex_pupil_amplitude_6510mm_1024pix.fits"
LS_FRAC = 0.85
LS_OBSCURATION_RATIO = 0.15
MAX_ITERS = 12_000
# --- 


# Set up the bandpass
half_bw = BANDWIDTH / 2 / 100
band = np.linspace(WVL * (1-half_bw), WVL * (1 + half_bw), NWVLS)

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
focal_plane_mask = annular_mask(iss, FPM_IWA, FPM_OWA)
dh = annular_mask(iss, IWA, OWA)
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

aplc = APLCOptimizer(amp = aperture - noisy,
                     amp_dx=pupil_dx,
                     efl=EFL,
                     wvl=WVL,
                     basis=None,
                     dark_hole=dh,
                     dh_target=0, # allows for specific contrast targeting, 0 just means "make it dark pls"
                     dh_dx=img_dx,
                     fpm=focal_plane_mask,
                     ls=ls_mask)

throughput = ThroughputOptimizer(amp=aperture-noisy,
                                 wvl=WVL,
                                 basis=None,
                                 ls=ls_mask,
                                 relative_weight=1e-6)

aplc.set_optimization_method(zonal=True)
throughput.set_optimization_method(zonal=True)

optlist = [aplc, throughput]

# optimization wrapper that sums the gradients and objective functions
opt_contrast_throughput = APLCWrapper(optlist=optlist)

# starting guess is a filled aperture
x0 = tnp.ones(aplc.amp.get().shape, dtype=float)[aplc.amp_select.get()]

# initialize the optimizer with box constraints
opt = F77LBFGSB(opt_contrast_throughput.fg, x0,
                memory=5, upper_bounds=tnp.ones(x0.shape),
                lower_bounds=tnp.zeros(x0.shape))
opt.iprint = 0

# some timing
t1 = time.perf_counter()
for _ in tqdm(range(MAX_ITERS)):
    opt.step()

print(f"Time to Optimizer for {MAX_ITERS}")
print(time.perf_counter() - t1)

newmask = aplc.amp
newmask[aplc.amp_select] = opt.x

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
plt.title('Pupil Apodizer')
if np.__name__ == "cupy":
    plt.imshow(newmask.get(),cmap='gray')
else:
    plt.imshow(newmask, cmap='gray')
plt.colorbar()
plt.subplot(133)
plt.title('Lyot Stop Field')
if np.__name__ == "cupy":
    plt.imshow(ls_mask.get()*(np.abs(aplc.c).get()),cmap='inferno')
else:
    plt.imshow(ls_mask*(np.abs(aplc.c)),cmap='inferno')
plt.colorbar()
plt.show()
