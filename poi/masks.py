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

def inner_core_mask(iss, iwa):

    x, y = coordinates.make_xy_grid(iss.N, dx=iss.dx)
    r, t = coordinates.cart_to_polar(x, y)
    iwa = iwa * iss.lamD
    mask = geometry.circle(iwa, r)

    return mask


def knife_edge_mask(iss, iwa):
    x, y = coordinates.make_xy_grid(iss.N, dx=iss.dx)
    iwa = iwa * iss.lamD
    mask = x > iwa

    return mask


def circular_mask(iss, iwa):
    x, y = coordinates.make_xy_grid(iss.N, dx=iss.dx)
    r, t = coordinates.cart_to_polar(x, y)
    iwa = iwa * iss.lamD
    mask = r > iwa

    return mask


def annular_mask(iss, iwa, owa, theta_min=None, theta_max=None):
    x, y = coordinates.make_xy_grid(iss.N, dx=iss.dx)
    r, t = coordinates.cart_to_polar(x, y)
    iwa = iwa * iss.lamD
    owa = owa * iss.lamD
    mask = r > iwa
    mask[r > owa] = 0
    
    if theta_min != None and theta_max != None:
        mask[t < np.radians(theta_min)] = 0
        mask[t > np.radians(theta_max)] = 0

    return mask


def lyot_mask(pupil_npix, pupil_dx, frac, obscuration_ratio=0.0):

    x, y = coordinates.make_xy_grid(pupil_npix, dx=pupil_dx)
    r, t = coordinates.cart_to_polar(x, y)
    rnorm = r / (r.max() * np.sqrt(2))
    ls = np.zeros_like(x)
    ls[rnorm < frac/2] = 1
    ls[rnorm < frac/2 * obscuration_ratio] = 0

    return ls
