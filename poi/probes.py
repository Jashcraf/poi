from prysm.mathops import np
import scipy


def create_sinc_probe(Nacts, amp, probe_radius, probe_phase=0, offset=(0,0), bad_axis='x'):
    """create sinc probe. Taken from uasal/lina, written by Kian Milani

    https://github.com/uasal/lina/blob/main/lina/utils.py

    Parameters
    ----------
    Nacts : int
        number of actuators across the DM
    amp : float
        probe amplitude, DM units
    probe_radius : float
        _description_
    probe_phase : float, optional
        phase offset to apply to probe, by default 0
    offset : tuple, optional
        offset from DM center to place probe, by default (0,0)
    bad_axis : str, optional
        _description_, by default 'x'

    Returns
    -------
    _type_
        _description_
    """
    
    xacts = np.arange( -(Nacts-1)/2, (Nacts+1)/2 )/Nacts - np.round(offset[0])/Nacts
    yacts = np.arange( -(Nacts-1)/2, (Nacts+1)/2 )/Nacts - np.round(offset[1])/Nacts

    Xacts, Yacts = np.meshgrid(xacts,yacts)

    if bad_axis=='x': 
        fX = 2*probe_radius
        fY = probe_radius
        omegaY = probe_radius/2
        probe_commands = amp * np.sinc(fX*Xacts)*np.sinc(fY*Yacts) * np.cos(2*np.pi*omegaY*Yacts + probe_phase)

    elif bad_axis=='y': 
        fX = probe_radius
        fY = 2*probe_radius
        omegaX = probe_radius/2
        probe_commands = amp * np.sinc(fX*Xacts)*np.sinc(fY*Yacts) * np.cos(2*np.pi*omegaX*Xacts + probe_phase) 

    if probe_phase == 0:
        f = 2*probe_radius
        probe_commands = amp * np.sinc(f*Xacts)*np.sinc(f*Yacts)
        
    return probe_commands


def create_annular_focal_plane_mask(npsf, psf_pixelscale, 
                                    inner_radius, outer_radius, 
                                    edge=None,
                                    shift=(0,0), 
                                    rotation=0,
                                    plot=False):
    x = (np.linspace(-npsf/2, npsf/2-1, npsf) + 1/2)*psf_pixelscale
    x,y = np.meshgrid(x,x)
    r = np.hypot(x, y)
    mask = (r < outer_radius) * (r > inner_radius)
    if edge is not None: mask *= (x > edge)
        
    return mask


def create_fourier_modes(dm_mask, npsf, psf_pixelscale_lamD, iwa, owa, 
                         rotation=0, 
                         fourier_sampling=0.75,
                         which='both', 
                         return_fs=False,
                         plot=False,
                         ):
    
    Nact = dm_mask.shape[0]
    nfg = int(np.round(npsf * psf_pixelscale_lamD/fourier_sampling))
    if nfg%2==1: nfg += 1
    yf, xf = (np.indices((nfg, nfg)) - nfg//2 + 1/2) * fourier_sampling
    fourier_cm = create_annular_focal_plane_mask(nfg, fourier_sampling, iwa-fourier_sampling, owa+fourier_sampling, edge=iwa-fourier_sampling, rotation=rotation)
    ypp, xpp = (np.indices((Nact, Nact)) - Nact//2 + 1/2)

    sampled_fs = np.array([xf[fourier_cm], yf[fourier_cm]]).T
    
    fourier_modes = []
    for i in range(len(sampled_fs)):
        fx = sampled_fs[i,0]
        fy = sampled_fs[i,1]
        if which=='both' or which=='cos':
            fourier_modes.append( dm_mask * np.cos(2 * np.pi * (fx*xpp + fy*ypp)/Nact) )
        if which=='both' or which=='sin':
            fourier_modes.append( dm_mask * np.sin(2 * np.pi * (fx*xpp + fy*ypp)/Nact) )
            
    if return_fs:
        return np.array(fourier_modes), sampled_fs
    else:
        return xp.array(fourier_modes)
    

def create_fourier_probes(dm_mask, npsf, psf_pixelscale_lamD, iwa, owa, 
                          rotation=0, 
                          fourier_sampling=0.75, 
                          shifts=None, nprobes=2,
                          use_weighting=False, 
                          plot=False,
                          ): 
    Nact = dm_mask.shape[0]
    cos_modes, fs = create_fourier_modes(dm_mask, npsf, psf_pixelscale_lamD, iwa, owa, rotation,
                                        fourier_sampling=fourier_sampling, 
                                        return_fs=True,
                                        which='cos',
                                        )
    sin_modes = create_fourier_modes(dm_mask, npsf, psf_pixelscale_lamD, iwa, owa, rotation,
                                    fourier_sampling=fourier_sampling, 
                                    which='sin',
                                    )
    nfs = fs.shape[0]

    probes = np.zeros((nprobes, Nact, Nact))
    if use_weighting:
        fmax = np.max(np.sqrt(fs[:,0]**2 + fs[:,1]**2))
        sum_cos = 0
        sum_sin = 0
        for i in range(nfs):
            f = np.sqrt(fs[i][0]**2 + fs[i][1]**2)
            weight = f/fmax
            sum_cos += weight*cos_modes[i]
            sum_sin += weight*sin_modes[i]
        sum_cos = sum_cos
        sum_sin = sum_sin
    else:
        sum_cos = cos_modes.sum(axis=0)
        sum_sin = sin_modes.sum(axis=0)
    
    # nprobes=2 will give one probe that is purely the sum of cos and another that is the sum of sin
    cos_weights = np.linspace(1,0,nprobes)
    sin_weights = np.linspace(0,1,nprobes)
    
    shifts = [(0,0)]*nprobes if shifts is None else shifts

    for i in range(nprobes):
        probe = cos_weights[i]*sum_cos + sin_weights[i]*sum_sin
        probe = scipy.ndimage.shift(probe, (shifts[i][1], shifts[i][0]))
        probes[i] = probe/np.max(probe)

    return probes