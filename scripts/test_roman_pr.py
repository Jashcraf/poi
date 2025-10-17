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
from poi.phase_retrieval import ADPhaseRetrieval, ParallelADPhaseRetrieval

# Load the pupil
NMODES = 37
LAMBDA_M = 550e-9
WAVE_MAG = 100
PHASE_SCALE = 1 # Artificially inflate Jones pupil elements
IMG_DX = 1
IMG_NPIX = 256
CGISIM_PATH = Path.home() / "Downloads/roman_preflight_proper_public_v2.0.1_python/roman_preflight_proper/preflight_data/hlc_20190210b"
POLABS_PATH = Path.home() / "Downloads/roman_preflight_proper_public_v2.0.1_python/roman_preflight_proper/preflight_data/pol"
defocus_waves = [0, 0.5, 1]
polarizer_angles = [0, 45, 90, 135]
stokes_vectors = [
    np.array([1, 0, 0, 0]),
]
tol = 1e-20


# Load the Jones pupils
roman_pupil = fits.getdata(CGISIM_PATH / "pupil.fits")
roman_pupil_amplitude = fits.getdata(POLABS_PATH / "preflight_pol_amp.fits")
roman_pupil_phase = fits.getdata(POLABS_PATH / "preflight_pol_pha.fits")
roman_pupil_hdu = fits.open(POLABS_PATH / "preflight_pol_amp.fits")

polpth = str(POLABS_PATH / "preflight_pol")
amp_j22, phs_j22 = polab(polpth, LAMBDA_M, roman_pupil.shape[0], condition=-4)
amp_j12, phs_j12 = polab(polpth, LAMBDA_M, roman_pupil.shape[0], condition=-3)
amp_j11, phs_j11 = polab(polpth, LAMBDA_M, roman_pupil.shape[0], condition=5)
amp_j21, phs_j21 = polab(polpth, LAMBDA_M, roman_pupil.shape[0], condition=6)

kvec = 2 * np.pi / LAMBDA_M
ipdb.set_trace()

jones_pupil = np.array([
    [amp_j11 * np.exp(1j * kvec * phs_j11*PHASE_SCALE), amp_j12 * np.exp(1j * kvec * phs_j12 * PHASE_SCALE)],
    [amp_j21 * np.exp(1j * kvec * phs_j21*PHASE_SCALE), amp_j22 * np.exp(1j * kvec * phs_j22 * PHASE_SCALE)],
])

fig, ax = plt.subplots(ncols=4, nrows=2)
for i in range(2):
    for j in range(2):

        J = jones_pupil[i, j] / roman_pupil
        ax[i, j].imshow(np.abs(J), vmin=0.98, vmax=1, cmap="inferno")
        ax[i, j+2].imshow(np.angle(J) / roman_pupil, vmin=-LAMBDA_M/WAVE_MAG, vmax=LAMBDA_M/WAVE_MAG, cmap="RdBu_r")

# Convert to prysm-friendly units
LAMBDA_M *= 1e6
EPD = 2400
EFL = 20e3

# Set up the defocus polynomial
def create_defocus_aberration(defocus_waves, Npup=amp_j11.shape[0]):

    x, y = make_xy_grid(Npup, diameter=EPD)
    r, t = cart_to_polar(x, y)
    r_z = r / (EPD / 2)
    defocus_polynomial = hopkins(0, 2, 0, r_z, t, 0)
    defocus_aberration = 2 * np.pi * defocus_polynomial * defocus_waves

    return defocus_aberration

# Construct the PSF

# Init the phase retrieval
x0 = np.random.random(NMODES)

# Construct a Zernike basis
x, y = make_xy_grid(roman_pupil.shape, diameter=2)
r, t = cart_to_polar(x, y)

nms = [noll_to_nm(i) for i in range(2, NMODES+2)]
basis = list(zernike_nm_sequence(nms, r, t))
masked_basis = [b * roman_pupil for b in basis]


# CONSTRUCT THE DATASET
stokes_images = []

for stokes in stokes_vectors:
    
    polarizer_images = []

    for polang in polarizer_angles:
        
        defocused_images = []
        
        for defocus in defocus_waves:
            
            # Construct the defocus aberration 
            defocus_phase = create_defocus_aberration(defocus)
            defocus_phasor = np.exp(1j * defocus_phase)

            # Construct the polarizer
            # Minus 45 comes from the basis of the Jones pupil being rotated into the +/-45 basis
            pol = linear_polarizer(theta=np.radians(polang - 45))
            
            # Init image
            amplitude_response_mat = np.zeros([IMG_NPIX, IMG_NPIX, 2, 2], dtype=np.complex128)
            
            # Generate amplitude response matrix
            for i in range(2):
                for j in range(2):

                    psf = focus_fixed_sampling(
                            wavefunction=jones_pupil[i, j] * roman_pupil * defocus_phasor,
                            input_dx=EPD/jones_pupil[i, j].shape[0],
                            prop_dist=EFL,
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
            defocused_images.append(image)
        
        # Append list of defocused images to list of polarizer images
        polarizer_images.append(defocused_images)
    
    # Append list of Stokes vectors observed to list of polarizer images
    stokes_images.append(polarizer_images)

# Run a bunch of individual phase retrieval experiments

# For each stokes vector observed
for pol_list in stokes_images:
    
    plt.figure()
    # For each polarizer angle
    for defocus_list in pol_list:
    
        optlist = []
        
        # For each defocus position
        for image, defocus in zip(defocus_list, defocus_waves):
            adpr = ADPhaseRetrieval(
                amp=roman_pupil,
                amp_dx=EPD / roman_pupil.shape[0],
                efl=EFL, # TODO: Check this
                wvl=LAMBDA_M,
                basis=basis,
                target=image,
                img_dx=IMG_DX,
                defocus_waves=defocus,
                initial_phase=None,
            )

            optlist.append(adpr)
        

        # Focus-diverse phase retrieval
        pzad_list = ParallelADPhaseRetrieval(optlist)
        results = minimize(pzad_list.fg, x0, jac=True, method="L-BFGS-B",
                           options={"maxiter": 100, "ftol":tol, "gtol":tol})
        print(results.message)
        phi = np.tensordot(masked_basis, results.x, axes=(0, 0))
        phi *= roman_pupil

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
