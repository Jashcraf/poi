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
from prysm.polynomials import noll_to_nm, hopkins, zernike_nm_seq as zernike_nm_sequence
from prysm.propagation import focus_fixed_sampling
from prysm.x.polarization import linear_polarizer, jones_to_mueller

# Pound the poi
from poi.phase_retrieval import PZPhaseRetrieval, ParallelADPhaseRetrieval

# Load the pupil
NMODES = 37
LAMBDA_M = 550e-9
WAVE_MAG = 100
IMG_DX = 1
IMG_NPIX = 256
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

PHASE_SCALE = 10
kvec = 2 * np.pi / LAMBDA_M

jones_pupil = np.array([
    [amp_j11 * np.exp(1j * kvec * phs_j11*PHASE_SCALE), amp_j12 * np.exp(1j * kvec * phs_j12 * PHASE_SCALE)],
    [amp_j21 * np.exp(1j * kvec * phs_j21*PHASE_SCALE), amp_j22 * np.exp(1j * kvec * phs_j22 * PHASE_SCALE)],
])

average_phase = 0
fig, ax = plt.subplots(ncols=4, nrows=2)
for i in range(2):
    for j in range(2):

        J = jones_pupil[i, j] / roman_pupil
        ax[i, j].imshow(np.abs(J), vmin=0.98, vmax=1, cmap="inferno")
        ax[i, j+2].imshow(np.angle(J) / roman_pupil, vmin=-LAMBDA_M/WAVE_MAG, vmax=LAMBDA_M/WAVE_MAG, cmap="RdBu_r")
        average_phase += np.angle(J)

# Get the average retardation
plt.figure()
plt.title("Average Phase solution")
plt.imshow(average_phase / roman_pupil, cmap="RdBu_r")
plt.colorbar()

# Convert to prysm-friendly units
LAMBDA_M *= 1e6

# Set up the defocus polynomial
def create_defocus_aberration(defocus_waves, Npup=amp_j11.shape[0]):

    x, y = make_xy_grid(Npup, diameter=2400)
    r, t = cart_to_polar(x, y)
    r_z = r / (2400 / 2)
    defocus_polynomial = hopkins(0, 2, 0, r_z, t, 0)
    defocus_aberration = 2 * np.pi * defocus_polynomial * defocus_waves

    return defocus_aberration

# Construct the PSF

# Init the vector phase retrieval
x0 = np.random.random(4 * NMODES) * 1e-10

# Construct a Zernike basis
x, y = make_xy_grid(roman_pupil.shape, diameter=2)
r, t = cart_to_polar(x, y)

nms = [noll_to_nm(i) for i in range(2, NMODES+2)]
basis = list(zernike_nm_sequence(nms, r, t))
masked_basis = [b * roman_pupil for b in basis]

defocus_waves = [0, 3]
polarizer_angles = [0]
stokes_vectors = [
    np.array([1, 0, 0, 0]),
]
optlist = []
defocused_images = []
for stokes in stokes_vectors:
    for polang in polarizer_angles:
        for defocus in defocus_waves:
            
            # Construct the defocus aberration 
            defocus_phase = create_defocus_aberration(defocus)
            defocus_phasor = np.exp(1j * defocus_phase)

            # Construct the polarizer
            pol = linear_polarizer(theta=np.radians(polang))
            pol = np.eye(2)
            
            # Init image
            amplitude_response_mat = np.zeros([IMG_NPIX, IMG_NPIX, 2, 2], dtype=np.complex128)
            
            # Generate amplitude response matrix
            for i in range(2):
                for j in range(2):

                    psf = focus_fixed_sampling(
                            wavefunction=jones_pupil[i, j] * roman_pupil * defocus_phasor,
                            input_dx=2400/jones_pupil[i, j].shape[0],
                            prop_dist=20e3,
                            wavelength=LAMBDA_M,
                            output_dx=IMG_DX,
                            output_samples=IMG_NPIX
                        )

                    amplitude_response_mat[..., i, j] = psf
            
            # Multiply by polarizer
            amplitude_response_mat = pol @ amplitude_response_mat

            # Convert to mueller matrix
            mueller_psm = jones_to_mueller(amplitude_response_mat)

            # Get image via stokes vector
            Sout = (mueller_psm @ stokes[..., None])[..., 0]
            
            # Final image comes from first element of stokes vector
            image = Sout[..., 0]
            
            pzad = PZPhaseRetrieval(
                amp=roman_pupil,
                amp_dx=2400 / roman_pupil.shape[0],
                efl=20e3, # TODO: Check this
                wvl=LAMBDA_M,
                basis=basis,
                target=image,
                img_dx=IMG_DX,
                defocus_waves=-defocus,
                initial_phase=None,
                stokes=stokes,
                waveplate_angle=0,
                polarizer_angle=polang
            )

            optlist.append(pzad)

pzad_list = ParallelADPhaseRetrieval(optlist)

f, g = pzad.fg(x0)
tol = 1e-50
results = minimize(pzad_list.fg, x0, jac=True, method="L-BFGS-B",
                   options={"maxiter": 100, "ftol":tol, "gtol":tol})
print(results)

# Construct Jones pupil from results
r_xx = results.x[0*NMODES : 1*NMODES]
r_xy = results.x[1*NMODES : 2*NMODES]
r_yx = results.x[2*NMODES : 3*NMODES]
r_yy = results.x[3*NMODES : 4*NMODES]

# i_xx = results.x[4*NMODES : 5*NMODES]
# i_xy = results.x[5*NMODES : 6*NMODES]
# i_yx = results.x[6*NMODES : 7*NMODES]
# i_yy = results.x[7*NMODES : 8*NMODES]

c_xx = r_xx # + 1j*i_xx
c_xy = r_xy # + 1j*i_xy
c_yx = r_yx # + 1j*i_yx
c_yy = r_yy # + 1j*i_yy

phi_xx = np.tensordot(masked_basis, c_xx, axes=(0, 0))
phi_xy = np.tensordot(masked_basis, c_xy, axes=(0, 0))
phi_yx = np.tensordot(masked_basis, c_yx, axes=(0, 0))
phi_yy = np.tensordot(masked_basis, c_yy, axes=(0, 0))

kvec = 2 * np.pi / LAMBDA_M

Jxx = np.exp(1j * phi_xx)
Jxy = np.exp(1j * phi_xy)
Jyx = np.exp(1j * phi_yx)
Jyy = np.exp(1j * phi_yy)

Jones_result = np.array([
    [Jxx, Jxy],
    [Jyx, Jyy]
])

Jones_result *= roman_pupil

f, g = pzad_list.fg(results.x)

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

for idx in [0, -1]:
    plt.figure(figsize=[10, 5])
    plt.subplot(121)
    plt.title("Reference PSF")
    plt.imshow(pzad_list.optlist[idx].D, norm=LogNorm())
    plt.colorbar()
    plt.subplot(122)
    plt.title("Model PSF")
    plt.imshow(pzad_list.optlist[idx].E, norm=LogNorm())
    plt.colorbar()

plt.figure()
for opt in pzad_list.optlist:
    plt.plot(opt.cost, label=f"pol={opt.polarizer_angle}, defocus={opt.defocus}")

plt.ylabel("Mean Squared Error")
plt.xlabel("Iteration")
plt.yscale("log")
plt.show()
