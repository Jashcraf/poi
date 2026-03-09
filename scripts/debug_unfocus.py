import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# The prysm stuff
from prysm.mathops import np
from prysm.fttools import MatrixDFTExecutor
from prysm.coordinates import make_xy_grid, cart_to_polar
from prysm.geometry import circle

# The test functions
from prysm.propagation import (
    focus_fixed_sampling,
    to_fpm_and_back,
    focus_fixed_sampling_backprop,
    to_fpm_and_back_backprop
)

# Define some optical system parameters
EPD = 25.4     # mm
EFL = EPD * 40 # mm
WVL = 1.       # um
NIMG = 256
NPUP = 256
Q = 4

img_dx = WVL * (EFL/EPD) / Q
pupil_dx = EPD / NPUP
img_dx_lamD = 1/Q

# Set up the masks
x, y = make_xy_grid(NPUP, diameter=EPD)
r, t = cart_to_polar(x, y)
pupil_mask = circle(EPD / 2, r)
lyot_mask = circle(EPD / 2 * 0.9, r)

u, v = make_xy_grid(NIMG, diameter=img_dx_lamD * NIMG)
rho, phi = cart_to_polar(u, v)
focal_mask = 1 - circle(3, rho)

masks = [pupil_mask, focal_mask, lyot_mask]
plt.figure(figsize=[10, 3])
for i, mask in enumerate(masks):
    plt.subplot(1, 3, i+1)
    plt.imshow(mask, cmap="gray")
    plt.colorbar()

# Set up forward propagation
to_lyot, to_fpm, _ = to_fpm_and_back(
    wavefunction=pupil_mask.astype(np.float64),
    dx=pupil_dx,
    efl=EFL,
    wavelength=WVL,
    fpm=focal_mask.astype(np.float64),
    fpm_dx=img_dx,
    return_more=True
)

to_coro = focus_fixed_sampling(
    wavefunction=to_lyot * lyot_mask.astype(np.float64),
    input_dx=EPD / NPUP,
    prop_dist=EFL,
    wavelength=WVL,
    output_dx=img_dx,
    output_samples=NIMG
)

fields = [np.abs(pupil_mask), np.abs(to_fpm)**2, np.abs(to_lyot)**2, np.abs(to_coro)**2]
labels = ["Pupil", "Focal Plane", "Lyot Plane", "Coronagraphic Plane"]
plt.figure(figsize=[12, 3])
plt.suptitle("Forward Model Fields")
for i, field in enumerate(fields):
    plt.subplot(1, 4, i+1)
    plt.title(labels[i])
    plt.imshow(field, cmap="inferno", norm=LogNorm())
    plt.colorbar(label="Log Irradiance")

# Now the backpropagation
from_coro = focus_fixed_sampling_backprop(
    wavefunction=to_coro,
    input_dx=pupil_dx,
    prop_dist=EFL,
    wavelength=WVL,
    output_dx=img_dx,
    output_samples=NPUP
)

from_lyot, from_fpm, _ = to_fpm_and_back_backprop(
    wavefunction=from_coro * lyot_mask,
    dx=pupil_dx,
    efl=EFL,
    wavelength=WVL,
    fpm=focal_mask.astype(np.float64),
    fpm_dx=img_dx,
    return_more=True
)

fields = [np.abs(from_lyot)**2, np.abs(from_fpm)**2, np.abs(from_coro)**2, np.abs(to_coro)**2]
plt.figure(figsize=[12, 3])
plt.suptitle("Backprop Model Fields")
for i, field in enumerate(fields):
    plt.subplot(1, 4, i+1)
    plt.title(labels[i]+" grad")
    plt.imshow(field, cmap="inferno", norm=LogNorm())
    plt.colorbar(label="Log Irradiance")
plt.show()
