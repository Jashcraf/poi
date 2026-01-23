from prysm.mathops import np, fft
from prysm import coordinates, geometry

class ImgSamplingSpec:
    """Specification for image plane sampling.
    Taken from the equivalent class at github.com/brandondube/dygdug
    """
    def __init__(self, N, dx, lamD):
        self.N = N
        self.dx = dx
        self.lamD = lamD

    @classmethod
    def from_N_lamD_px_per_lamD(cls, N, lamD, px_per_lamD):
        dx = lamD/px_per_lamD
        return cls(N=N, dx=dx, lamD=lamD)

def _make_symmetric_xy(N, dx, shift=False):
    """
    Controling the precise centering of prysm's xy grids
    """
    x, y = coordinates.make_xy_grid(N, dx=dx)

    if shift:
        shift = dx / 2
        x += shift
        y += shift

    return x, y

def inner_core_mask(iss, iwa):

    x, y = _make_symmetric_xy(iss.N, dx=iss.dx)
    r, t = coordinates.cart_to_polar(x, y)
    iwa = iwa * iss.lamD
    mask = geometry.circle(iwa, r)

    return mask


def knife_edge_mask(iss, iwa):
    x, y = _make_symmetric_xy(iss.N, dx=iss.dx)
    iwa = iwa * iss.lamD
    mask = x > iwa

    return mask


def circular_mask(iss, iwa):
    x, y = _make_symmetric_xy(iss.N, dx=iss.dx)
    r, t = coordinates.cart_to_polar(x, y)
    iwa = iwa * iss.lamD
    mask = r > iwa

    return mask


def annular_mask(iss, iwa, owa, theta_min=None, theta_max=None, shift=False):
    x, y = _make_symmetric_xy(iss.N, dx=iss.dx, shift=shift)
    r, t = coordinates.cart_to_polar(x, y)
    iwa = iwa * iss.lamD
    owa = owa * iss.lamD

    mask = (r > iwa) & (r <= owa)

    t_norm = t % (2 * np.pi)

    if theta_min is not None and theta_max is not None:

        # Normalize angles to [0, 2π]
        t_norm = t % (2 * np.pi)
        
        theta_min_rad = np.radians(theta_min) % (2 * np.pi)
        theta_max_rad = np.radians(theta_max) % (2 * np.pi)
        
        # Create mask for primary wedge
        if theta_max_rad < theta_min_rad:
            # Range wraps around
            angular_mask = (t_norm >= theta_min_rad) | (t_norm <= theta_max_rad)
        else:
            # Normal case
            angular_mask = (t_norm >= theta_min_rad) & (t_norm <= theta_max_rad)
        
        # Create mask for opposite wedge (180° rotated)
        t_opposite = (t_norm + np.pi) % (2 * np.pi)
        if theta_max_rad < theta_min_rad:
            angular_mask_opposite = (t_opposite >= theta_min_rad) | (t_opposite <= theta_max_rad)
        else:
            angular_mask_opposite = (t_opposite >= theta_min_rad) & (t_opposite <= theta_max_rad)
        
        # Combine both wedges
        angular_mask = angular_mask | angular_mask_opposite
        
        # Combine radial and angular masks
        mask = mask & angular_mask
    
    return mask

def lyot_mask(pupil_npix, pupil_dx, frac, obscuration_ratio=0.0, shift=False):

    x, y = _make_symmetric_xy(pupil_npix, dx=pupil_dx, shift=shift)
    r, t = coordinates.cart_to_polar(x, y)
    rnorm = r / (r.max() * np.sqrt(2))
    ls = np.zeros_like(x)
    ls[rnorm < frac/2] = 1
    ls[rnorm < frac/2 * obscuration_ratio] = 0

    return ls
