import numpy as np

def calc_log_dust(tau):
    '''
    Calculate the dust surface density as in Brinchmann et al. 2013
    :param tau: Optical depth at 5500 Angstrom
    :return: Log of dust surface density
    '''
    return np.log10(0.2 * tau)

def calc_log_gas(logZ, xi, tau):
    '''
    Calculate the gas surface density as in Brinchmann et al. 2013
    :param logZ: Log metallicity in units of solar metallicity
    :param xi: Dust-to-metal ratio
    :param tau: Optical depth at 5500 Angstrom
    :return: Log of gas surface density
    '''
    Zsun = 0.0142 # Asplund (2009) photospheric mass fraction
    return np.log10(0.2 * tau/(xi * 10**logZ * Zsun))
