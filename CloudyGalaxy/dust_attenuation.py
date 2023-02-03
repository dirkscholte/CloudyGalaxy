import numpy as np


def transmission_function(lambda_, logtau, n=-1.3):
    """
    Function to calculate the transmission function. As in Charlot and Fall (2000)
    :param lambda_: Wavelength values of the spectrum bins in Angstrom
    :param logtau: Log optical depth at 5500 Angstrom
    :param n: Exponent of power law. Default is -1.3 as is appropriate for birth clouds (-0.7 for general ISM).
    :return: Transmission function for each bin in the spectrum
    """
    lambda_ = np.array(lambda_)
    return np.exp(-(10**logtau) * (lambda_ / 5500) ** n)
