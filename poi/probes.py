from prysm.mathops import np


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