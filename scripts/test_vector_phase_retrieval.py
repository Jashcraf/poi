import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from matplotlib.colors import LogNorm
from pathlib import Path
import ipdb
from polmap import polmap, polab
from scipy.optimize import minimize

# The prysm imports
from prysm.coordinates import make_xy_grid, cart_to_polar
from prysm.polynomials import noll_to_nm, zernike_nm_seq as zernike_nm_sequence
from prysm.propagation import focus_fixed_sampling

# Pound the poi
from poi.phase_retrieval import PZPhaseRetrieval, ParallelADPhaseRetrieval

# Load the pupil
NMODES = 37
LAMBDA_M = 550e-9
WAVE_MAG = 100
IMG_DX = 1 
CGISIM_PATH = Path.home() / "Downloads/roman_preflight_proper_public_v2.0.1_python/roman_preflight_proper/preflight_data/hlc_20190210b"
POLABS_PATH = Path.home() / "Downloads/roman_preflight_proper_public_v2.0.1_python/roman_preflight_proper/preflight_data/pol"
roman_pupil = fits.getdata(CGISIM_PATH / "pupil.fits")
roman_pupil_amplitude = fits.getdata(POLABS_PATH / "preflight_pol_amp.fits")
roman_pupil_phase = fits.getdata(POLABS_PATH / "preflight_pol_pha.fits")
roman_pupil_hdu = fits.open(POLABS_PATH / "preflight_pol_amp.fits")

# Load the Jones pupils
polpth = str(POLABS_PATH / "preflight_pol")
amp_j22, phs_j22 = polab(polpth, LAMBDA_M, roman_pupil.shape[0], condition=-4)
amp_j12, phs_j12 = polab(polpth, LAMBDA_M, roman_pupil.shape[0], condition=-3)
amp_j11, phs_j11 = polab(polpth, LAMBDA_M, roman_pupil.shape[0], condition=5)
amp_j21, phs_j21 = polab(polpth, LAMBDA_M, roman_pupil.shape[0], condition=6)

jones_pupil = np.array([
    [amp_j11 * np.exp(1j * phs_j11), amp_j12 * np.exp(1j * phs_j12)],
    [amp_j21 * np.exp(1j * phs_j21), amp_j22 * np.exp(1j * phs_j22)],
])

fig, ax = plt.subplots(ncols=4, nrows=2)
for i in range(2):
    for j in range(2):

        J = jones_pupil[i, j] / roman_pupil
        ax[i, j].imshow(np.abs(J), vmin=0.98, vmax=1, cmap="inferno")
        ax[i, j+2].imshow(np.angle(J) / roman_pupil, vmin=-LAMBDA_M/WAVE_MAG, vmax=LAMBDA_M/WAVE_MAG, cmap="RdBu_r")

# Convert to prysm-friendly units
LAMBDA_M *= 1e6

# Construct the PSF
image = 0
for i in range(2):
    for j in range(2):
        
        psf = focus_fixed_sampling(
                wavefunction=jones_pupil[i, j] * roman_pupil,
                input_dx=2400/jones_pupil[i, j].shape[0],
                prop_dist=20e3,
                wavelength=LAMBDA_M,
                output_dx=IMG_DX,
                output_samples=128
            )

        image += np.abs(psf)**2
# Init the vector phase retrieval
x0 = np.random.random(8 * NMODES) / 100

# Construct a Zernike basis
x, y = make_xy_grid(roman_pupil.shape, diameter=2)
r, t = cart_to_polar(x, y)

nms = [noll_to_nm(i) for i in range(1, NMODES+1)]
basis = list(zernike_nm_sequence(nms, r, t))
masked_basis = [b * roman_pupil for b in basis]

polarizer_angles = [0, 45, 90, 135]
defocus_waves = [0, 3]
optlist = []

for polang in polarizer_angles:
    for defocus in defocus_waves:

        pzad = PZPhaseRetrieval(
            amp=roman_pupil,
            amp_dx=2400 / roman_pupil.shape[0],
            efl=20e3, # TODO: Check this
            wvl=LAMBDA_M,
            basis=basis,
            target=image,
            img_dx=IMG_DX,
            defocus_waves=defocus,
            initial_phase=None,
            stokes=np.array([1., 0., 0., 0.]),
            waveplate_angle=45,
            polarizer_angle=polang
        )

        optlist.append(pzad)

pzad_list = ParallelADPhaseRetrieval(optlist)

f, g = pzad.fg(x0)
tol = 1e-40
results = minimize(pzad_list.fg, x0, jac=True, method="L-BFGS-B",
                   options={"maxiters": 1000, "ftol":tol, "gtol":tol})
print(results)

# Construct Jones pupil from results
r_xx = results.x[0*NMODES : 1*NMODES]
r_xy = results.x[1*NMODES : 2*NMODES]
r_yx = results.x[2*NMODES : 3*NMODES]
r_yy = results.x[3*NMODES : 4*NMODES]

i_xx = results.x[4*NMODES : 5*NMODES]
i_xy = results.x[5*NMODES : 6*NMODES]
i_yx = results.x[6*NMODES : 7*NMODES]
i_yy = results.x[7*NMODES : 8*NMODES]

c_xx = r_xx + 1j*i_xx
c_xy = r_xy + 1j*i_xy
c_yx = r_yx + 1j*i_yx
c_yy = r_yy + 1j*i_yy

Jxx = np.tensordot(masked_basis, c_xx, axes=(0, 0))
Jxy = np.tensordot(masked_basis, c_xy, axes=(0, 0))
Jyx = np.tensordot(masked_basis, c_yx, axes=(0, 0))
Jyy = np.tensordot(masked_basis, c_yy, axes=(0, 0))
Jones_result = np.array([
    [Jxx, Jxy],
    [Jyx, Jyy]
])

fig, ax = plt.subplots(ncols=4, nrows=2)
fig.suptitle("Retrieved Jones Pupil")
for i in range(2):
    for j in range(2):

        J = Jones_result[i, j]
        im = ax[i, j].imshow(np.abs(J), cmap="inferno")
        div = make_axes_locatable(ax[i, j])
        cax = div.append_axes("right", size="7%", pad="2%")
        fig.colorbar(im, cax=cax)
        
        im = ax[i, j+2].imshow(np.angle(J), cmap="RdBu_r")
        div = make_axes_locatable(ax[i, j+2])
        cax = div.append_axes("right", size="7%", pad="2%")
        fig.colorbar(im, cax=cax)

plt.figure(figsize=[10, 5])
plt.subplot(121)
plt.title("Reference PSF")
plt.imshow(image, norm=LogNorm())
plt.colorbar()
plt.subplot(122)
plt.title("Model PSF")
plt.imshow(pzad.E, norm=LogNorm())
plt.colorbar()

plt.show()
