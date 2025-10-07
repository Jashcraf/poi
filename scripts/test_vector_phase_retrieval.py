import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from pathlib import Path
import ipdb
from polmap import polmap, polab
from poi.phase_retrieval import PZPhaseRetrieval

# The prysm imports
from prysm.coordinates import make_xy_grid, cart_to_polar
from prysm.polynomials import noll_to_nm, zernike_nm_seq as zernike_nm_sequence

# Load the pupil
NMODES = 12
LAMBDA_M = 550e-9
WAVE_MAG = 100
CGISIM_PATH = Path.home() / "Downloads/roman_preflight_proper_public_v2.0.1_python/roman_preflight_proper/preflight_data/hlc_20190210b"
POLABS_PATH = Path.home() / "Downloads/roman_preflight_proper_public_v2.0.1_python/roman_preflight_proper/preflight_data/pol"
roman_pupil = fits.getdata(CGISIM_PATH / "pupil.fits")
roman_pupil_amplitude = fits.getdata(POLABS_PATH / "preflight_pol_amp.fits")
roman_pupil_phase = fits.getdata(POLABS_PATH / "preflight_pol_pha.fits")
roman_pupil_hdu = fits.open(POLABS_PATH / "preflight_pol_amp.fits")

# Load the Jones pupils
plt.figure()
plt.imshow(roman_pupil, cmap="gray")
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


# Init the vector phase retrieval
x0 = np.random.random(8 * NMODES)

# Construct a Zernike basis
x, y = make_xy_grid(roman_pupil.shape, diameter=2)
r, t = cart_to_polar(x, y)

nms = [noll_to_nm(i) for i in range(1, NMODES+1)]
basis = list(zernike_nm_sequence(nms, r, t))
masked_basis = [b * roman_pupil for b in basis]

pzad = PZPhaseRetrieval(
    amp=roman_pupil,
    amp_dx=roman_pupil.shape[0] / 2.4,
    efl=20e3, # TODO: Check this
    wvl=0.55,
    basis=basis,
    target=0,
    img_dx=1,
    defocus_waves=0,
    initial_phase=None,
    stokes=np.array([1., 0., 0., 0.]),
    waveplate_angle=45,
    polarizer_angle=90
)

f, g = pzad.fg(x0)


plt.show()
