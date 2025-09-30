import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from pathlib import Path
import ipdb
from polmap import polmap, polab

# The prysm imports

# Load the pupil
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
amp, phs = polab(polpth, LAMBDA_M, roman_pupil.shape[0], condition=1)

plt.figure()
plt.subplot(121)
plt.imshow(amp / roman_pupil, cmap="inferno", vmin=0.9, vmax=1)
plt.colorbar()
plt.subplot(122)
plt.imshow(phs / roman_pupil, cmap="RdBu_r", vmin=-LAMBDA_M/WAVE_MAG, vmax=LAMBDA_M/WAVE_MAG)
plt.colorbar()
plt.show()

