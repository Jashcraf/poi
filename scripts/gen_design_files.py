# The regular stuff
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from tqdm import tqdm
from astropy.io import fits
from pathlib import Path
import time
import numpy as tnp
from scipy.ndimage import shift
import os

# The prysm stuff
from prysm.mathops import np, set_backend_to_cupy
from prysm.propagation import focus_fixed_sampling, unfocus_fixed_sampling
from prysm.fttools import MatrixDFTExecutor

# Available optimizers
from prysm.x.optym import (
    F77LBFGSB, # The one that works with Box constraints
)

# Poi local
from poi.masks import ImgSamplingSpec, inner_core_mask, annular_mask, lyot_mask
from poi.aplc_design import AmplitudeAPLC, APLCWrapper, ThroughputOptimizer
from poi.propagation import convolve_2d
from poi.cost_functions import (
    CoreThroughput,
    LogSumExp,
    MeanSquaredErrorLinear,
    MeanSquaredErrorQuadratic,
    PNorm
) 

# --- USER INPUT DESIGN PARAMS HERE
USE_GPU = True # Use GPU for the optimization
EPD = 24.4381  # milimeters
EFL = EPD * 40 # milimeters
WVL = 0.350 # microns
IMG_NPIX = (256)
IWA = 6
OWA = 20
AZMIN = -65 / 2 # Defines the angular extend of the dark zone
AZMAX = 65 / 2
BANDWIDTH = 10 # percent
NWVLS = 5
OVERSAMPLE = 4 # pix per lam/D
pth_to_aperture = Path.home() / "poi/hex_pupil_amplitude_6510mm_1024pix.fits"
pth_to_aperture = Path.home() / "poi/luvoir_b_pupil_512.0_shift_px_py.fits"
LS_FRAC = 0.95 # Fraction of the pupil radius to use for the Lyot stop
LS_OBSCURATION_RATIO = 0.0 # Ratio of the Lyot stop obscuration to the pupil radius
MAX_ITERS = 10_000
core_size = 0.7 # radius in lam/D
TARGET_CONTRAST = 1e-13
CONTRAST_RELATIVE_WEIGHT = 1 # 1e10 worked here, 1e7 too low for point-symmetric
THROUGHPUT_LOG_WEIGHT = 21.5
THROUGHPUT_RELATIVE_WEIGHT =  10 ** (-1 * THROUGHPUT_LOG_WEIGHT)
SAVE_DIR = Path.home() / "poi/Data"

if USE_GPU:
    # np switches from numpy to cupy
    set_backend_to_cupy()

# Configure save directory
from datetime import datetime
now = datetime.now()
filename = f"APLC_{THROUGHPUT_LOG_WEIGHT}_{now.strftime('%Y-%m-%d_%H-%M-%S')}"
SAVE_DIR = SAVE_DIR / filename
os.makedirs(SAVE_DIR, exist_ok=True)

tilt_lds = np.arange(0, OWA, 0.05)

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

if np.__name__ == "cupy":
    psf = (psf_scalar / psf_scalar.max()).get()
else:
    psf = psf_scalar / psf_scalar.max()


iss = ImgSamplingSpec(IMG_NPIX, lambd / OVERSAMPLE, lambd)
focal_plane_mask = annular_mask(iss, FPM_IWA, FPM_OWA, theta_min=AZMIN, theta_max=AZMAX)

dh = annular_mask(iss, IWA, OWA, theta_min=AZMIN, theta_max=AZMAX)

ls_mask = lyot_mask(PUPIL_NPIX, pupil_dx=pupil_dx, frac=LS_FRAC, obscuration_ratio=LS_OBSCURATION_RATIO, shift=True)
# aperture = lyot_mask(PUPIL_NPIX, pupil_dx=pupil_dx, frac=1.)

# aperture = shift_left(aperture)


optlist = []
for wave in band:

    # Set up cost function
    cost = MeanSquaredErrorQuadratic(target=TARGET_CONTRAST)
    # cost = LogSumExp(target=TARGET_CONTRAST, alpha=1e5)

    aplc = AmplitudeAPLC(amp=aperture,
                        amp_dx=pupil_dx,
                        efl=EFL,
                        wvl=wave,
                        dark_hole=dh,
                        dh_dx=img_dx,
                        fpm=focal_plane_mask,
                        ls=ls_mask,
                        weight=CONTRAST_RELATIVE_WEIGHT,
                        cost_function=cost,
                        symmetry="point")

    aplc.set_optimization_method(zonal=True)
    optlist.append(aplc)

# Set up CoreThroughput cost function
core_mask = inner_core_mask(iss, core_size)
core_throughput = CoreThroughput(target=0)

throughput = AmplitudeAPLC(amp=aperture,
                        amp_dx=pupil_dx,
                        efl=EFL,
                        wvl=wave,

                        # NOTE this is no longer a dark hole,
                        # but a PSF core window
                        dark_hole=core_mask, 
                        dh_dx=img_dx,
                        fpm=focal_plane_mask,
                        ls=ls_mask,
                        weight=THROUGHPUT_RELATIVE_WEIGHT,

                        # The cost function is now altered to max core throughput
                        cost_function=core_throughput,
                        symmetry="point",
                        include_fpm=False)

throughput = ThroughputOptimizer(amp=aperture,
                                  wvl=wave,
                                  basis=None,
                                  ls=ls_mask,
                                  relative_weight=THROUGHPUT_RELATIVE_WEIGHT,
                                  symmetry="point")

throughput.set_optimization_method(zonal=True)
optlist.append(throughput)


# optimization wrapper that sums the gradients and objective functions
opt_contrast_throughput = APLCWrapper(optlist=optlist)

plt.figure()
combo = np.float64(aplc.amp_select.get())
combo += np.float64(np.fliplr(aplc.amp_select).get())
combo += np.float64(np.flipud(aplc.amp_select).get())
combo += np.float64(np.fliplr(np.flipud(aplc.amp_select)).get())
plt.title("There shouldn't be a vertical or horizontal line")
plt.imshow(combo)
plt.colorbar()

# starting guess is a filled aperture
# If point_symmetric=True, this filters out the aperture
if np.__name__ == "cupy":
    x0 = tnp.ones(aplc.amp.get().shape, dtype=float)[aplc.amp_select.get()]
else:
    x0 = tnp.ones(aplc.amp.shape, dtype=float)[aplc.amp_select]

x0 -= tnp.random.random(x0.shape) / 1000


# Dry-run to debug
_, _ = opt_contrast_throughput.fg(x0)

# Just plot the focal plane
plt.figure()
plt.subplot(131)
plt.title("Re|Pupil Gradient| before optimization")
plt.imshow(tnp.real(opt_contrast_throughput.optlist[0].bbar.get()))
plt.colorbar()
plt.subplot(132)
plt.title("Difference in gradient Q1-Q2")
_opt = opt_contrast_throughput.optlist[0]
xbar = _opt.aplcbar * _opt.amp_select
xbar_flipped = np.fliplr(_opt.aplcbar * np.fliplr(_opt.amp_select))
plt.imshow((xbar - xbar_flipped).get(), cmap="RdBu_r")
plt.colorbar()
plt.subplot(133)
plt.title("Difference in gradient Q1-Q4")
xbar = _opt.aplcbar * _opt.amp_select
xbar_flipped = np.flipud(_opt.aplcbar * np.flipud(_opt.amp_select))
plt.imshow((xbar - xbar_flipped).get(), cmap="RdBu_r")
plt.colorbar()

knife_right = _opt.amp_select
knife_left = np.fliplr(_opt.amp_select)
b_right = _opt.b * knife_right
b_left = np.fliplr(_opt.b * knife_left)

focal_knife_right = np.ones_like(np.real(_opt.B))
focal_knife_right[:, :_opt.B.shape[0] // 2] = 0
focal_knife_left = np.fliplr(focal_knife_right)

B_right = np.abs(_opt.B)**2 * focal_knife_right
B_left = np.fliplr(np.abs(_opt.B)**2 * focal_knife_left)
B_right = np.abs(_opt.B)**2 # * focal_knife_right
B_left = np.fliplr(B_right)


# initialize the optimizer with box constraints
opt = F77LBFGSB(opt_contrast_throughput.fg, x0,
                memory=10, upper_bounds=tnp.ones(x0.shape),
                lower_bounds=tnp.zeros(x0.shape))
opt.iprint = 1

# some timing
t1 = time.perf_counter()

# This is in a try-except block because the optimizer will
# sometimes raise a StopIteration exception when it is done

# Does not appear to work :/
N_RELAXATIONS = 1

# Set up a gaussian kernel
npx = aperture.shape[0]
sigma = 1
xx = np.linspace(-npx//2, npx//2+1, npx)
xx, yy = np.meshgrid(xx, xx)
r = np.hypot(xx, yy)
kernel = np.exp(-0.5 * (r/sigma)**2)

for jj in range(N_RELAXATIONS):
    
    newmask = np.zeros_like(aplc.amp, dtype=float)
    newmask[aplc.amp_select] = opt.x
    
    plot_amp_select = aplc.amp_select.copy()
    plot_newmask = newmask.copy()

    if hasattr(plot_amp_select, "get"):
        plot_amp_select = plot_amp_select.get()
        plot_newmask = plot_newmask.get()

    if aplc.symmetry == "point":
        newmask += np.fliplr(newmask)
    
    elif aplc.symmetry == "quadrant":
        Q1 = np.copy(newmask)
        newmask += np.fliplr(Q1)
        newmask += np.flipud(Q1)
        newmask += np.fliplr(np.flipud(Q1))

    if hasattr(newmask, "get"):
        newmask = newmask.get()

    plt.figure(figsize=[12, 4])
    plt.title(f"Apodizer iteration = {jj}")
    plt.imshow(newmask, cmap="gray")
    plt.colorbar()

    try:
        for _ in tqdm(range(MAX_ITERS)):
            opt.step()
    except StopIteration:
        pass
    
    print(f"Time to Optimizer for {MAX_ITERS}")
    print(time.perf_counter() - t1)


newmask = np.zeros_like(aplc.amp, dtype=float)
newmask[aplc.amp_select] = opt.x

if aplc.symmetry == "point":
    newmask += np.fliplr(newmask)
elif aplc.symmetry == "quadrant":
    Q1 = np.copy(newmask)
    newmask += np.fliplr(Q1)
    newmask += np.flipud(Q1)
    newmask += np.fliplr(np.flipud(Q1))

# difference the left and right halves
knife_right = np.ones_like(aplc.amp_select)
knife_right[:, :newmask.shape[0] // 2] = 0
knife_left = np.fliplr(knife_right)

b_right = newmask * knife_right
b_left = np.flipud(newmask * knife_right)

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
            before_fpm *= fpm

        before_ls = unfocus_fixed_sampling(
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
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm
throughput = []

throughput_07 = []

# Get the contrast normalization
before, _, coro = prop_coro(newmask, focal_plane_mask, ls_mask, tilt=0, include_fpm=False, wave=band)
contrast_norm = before.max()

before, lyot_field, coro = prop_coro(newmask, focal_plane_mask, ls_mask, tilt=0, include_fpm=True, wave=band)
contrast_onax = coro / contrast_norm

# Lyot Stop plot
ax3.set_title('PSF Morphology')
if np.__name__ == "cupy":
    im = ax3.imshow(contrast_onax.get(), cmap='inferno', norm=LogNorm(vmax=1e-5, vmin=1e-11))
else:
    im = ax3.imshow(contrast_onax, cmap='inferno', norm=LogNorm(vmax=1e-5, vmin=1e-11))

# Clear ticks
ax3.set_xticks([])
ax3.set_xticklabels([])
ax3.set_yticks([])
ax3.set_yticklabels([])

# Set up colorbar
div = make_axes_locatable(ax3)
cax = div.append_axes("right", size="5%", pad=0.1)
fig.colorbar(im, cax=cax)


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
if hasattr(masked_contrast, "get"):
    masked_contrast = masked_contrast.get()
radial_profile, bins = azimuthal_average(masked_contrast, angle_range=[AZMIN, AZMAX])
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

# Clean up and save
mdft.clear()

if hasattr(aperture, "get"):
    aperture = aperture.get()

if hasattr(newmask, "get"):
    newmask = newmask.get()

if hasattr(focal_plane_mask, "get"):
    focal_plane_mask = focal_plane_mask.get()

if hasattr(ls_mask, "get"):
    ls_mask = ls_mask.get()

# Construct fits headers
hdu_aper = fits.PrimaryHDU(aperture.astype(tnp.float64))
hdu_apod = fits.PrimaryHDU(newmask.astype(tnp.float64))
hdu_fpm = fits.PrimaryHDU(focal_plane_mask.astype(tnp.float64))
hdu_lyot = fits.PrimaryHDU(ls_mask.astype(tnp.float64))

# Add sampling parameters to apodizer
#################00000000############
hdu_apod.header["NPIX"] = newmask.shape[0]
hdu_apod.header["DX"] = (EPD, "[mm]")
hdu_apod.header["F/#"] = 40
hdu_apod.header["EFL"] = (EFL, '[mm]')
hdu_apod.header["CEN WVL"] = (WVL, '[microns]')
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
hdu_fpm.header["DX"] = (1/OVERSAMPLE, "[lam/D/pix]")
hdu_fpm.header["NPIX"] = focal_plane_mask.shape[0]


hdu_apod.writeto(SAVE_DIR / f"apodizer_{WVL}cenwvl_{NWVLS}wvls_{BANDWIDTH}%.fits")
hdu_lyot.writeto(SAVE_DIR / f"lyot_stop.fits")
hdu_fpm.writeto(SAVE_DIR / f"fpm_{IWA}IWA_{OWA}OWA_{OVERSAMPLE}OS_{AZMIN}min_{AZMAX}max.fits")
hdu_aper.writeto(SAVE_DIR / f"aperture.fits")
