import numpy as np
from scipy.integrate import simps
from astropy import constants as const
from astropy import units as u
from astropy.units import cds

class GasStatsCL01:
    '''
    Calculates gas statistics using the Charlot and Longhetti (2001) parametrization.
    '''
    def __init__(self, logZ, logU, age=1e8, M_star=3e4, nH=30.0):
        '''
        :param logZ: Log metallicity in units of solar metallicity
        :param logU: Log ionization parameter
        :param age: age of the SSP in years
        :param M_star: Stellar mass of the source of ionizing radiation in solar masses
        :param nH: Hydrogen density in cm^-3
        :param lambda_: Wavelength values of the spectrum bins in Angstrom
        :param spec: Luminosity values of spectrum bins at age of SSP in erg*s^-1*AA^-1
        :param spec_0: Luminosity values of spectrum bins of zero-age SSP in erg*s^-1*AA^-1
        :param Q_0: Rate of ionizing photons at zero-age in s^-1
        :param Q: Current rate of ionizing photons in s^-1
        :param filling_factor: Fraction of the volume that contains gas
        '''
        self.logZ = logZ
        self.logU = logU
        self.age = age
        self.M_star = M_star
        self.nH = nH
        self.lambda_ = np.array([])
        self.spec = np.array([])
        self.spec_0 = np.array([])
        self.Q_0 = np.nan
        self.Q = np.nan
        self.filling_factor = np.nan

    def set_stellar_spectrum(self):
        '''
        Set internal parameters lambda_ spec, spec_0
        :return: Nothing, update internal parameters
        '''
        self.lambda_, self.spec = self.stellar_spectrum.load_spectrum(self.age, self.logZ)
        __          , self.spec_0 = self.stellar_spectrum.load_spectrum(0.0, self.logZ)

    def calc_Q(self, lambda_, spec):
        '''
        Calculate the rate of ionizing photons
        :param lambda_: Wavelength values of the spectrum bins in AA
        :param spec: Luminosity values of spectrum bins in erg*s^-1*AA^-1
        :return: Rate of ionizing photons in s^-1
        '''
        h = const.h.to(u.erg * u.s**-1).value # erg * s^-1
        c = const.c.to(cds.AA * u.s**-1).value # AA * s^-1

        mask = lambda_ <= 912.
        integral = simps(lambda_[mask] * spec[mask], lambda_[mask])
        return self.M_star / (h * c) * integral

    def Q_spectrum(self, zero_age=False):
        '''
        Calculate the rate of ionizing photons
        :param zero_age: Determine whether you want to return zero-age or current rate of ionizing photons.
        :return: Rate of ionizing photons.
        '''
        self.Q_0 = self.calc_Q(lambda_, spec_0)
        self.Q = self.calc_Q(lambda_, spec)
        if zero_age:
            return self.Q_0
        else:
            return self.Q

    def calc_filling_factor(self, U_0, Q_0, nH):
        '''
        Calculate the filling factor.
        :param U_0: Zero-age ionization parameter
        :param Q_0: Zero-age rate of ionizing photons
        :param nH: Hydrogen density in cm^-3
        :return: Filling factor
        '''
        c = const.c.to(u.cm * u.s**-1).value # cm * s^-1
        alpha_B = 2.59e-13  # cm^3 s^-1  Alpha Case B (not in astropy units)
        return ((4 / 3) * np.pi * c ** 3 / alpha_B ** 2 * U_0 ** 3 / (Q_0 * nH)) ** 0.5

    def filling_factor_spectrum(self):
        '''
        Calculate the filling factor for spectrum
        :return: Filling factor, update internal filling factor parameter
        '''
        Q_0 = self.Q_spectrum(zero_age=True)
        U_0 = 10 ** self.logU
        self.filling_factor = self.calc_filling_factor(U_0, Q_0, self.nH)
        return self.filling_factor

    def calc_all_parameters(self):
        '''
        Calculate all parameters.
        :return: Nothing, update internal parameters.
        '''
        self.set_stellar_spectrum()
        self.Q_spectrum()
        self.filling_factor_spectrum()


class GasStatsBy17:
    '''
    Calculates gas statistics using the Byler et al. (2017) parametrization.
    '''
    def __init__(self, stellar_spectrum, logZ, logU, age=1e6, nH=100.0, r_inner=1e19):
        '''
        :param stellar_spectrum: Instance of the StellarSpectrum class with the path to data set.
        :param logZ: Log metallicity in units of solar metallicity
        :param logU: Log ionization parameter
        :param age: age of the SSP in years
        :param nH: Hydrogen density in cm^-3
        :param r_inner: Inner radius of the gas cloud in cm
        :param lambda_: Wavelength values of the spectrum bins in AA
        :param spec: Luminosity values of spectrum bins at age of SSP in erg*s^-1*AA^-1
        :param spec_0: Luminosity values of spectrum bins of zero-age SSP in erg*s^-1*AA^-1
        :param Q_hat_0: Rate of ionizing photons from a zero-age SSP in s^-1
        :param Q_0: Rate of ionizing photons necessary to satisfy conditions in s^-1
        :param M_star: Stellar mass of the source of ionizing radiation in solar masses
        :param filling_factor: Fraction of the volume that contains gas
        '''
        self.stellar_spectrum = stellar_spectrum
        self.logZ = logZ
        self.logU = logU
        self.age = age
        self.nH = nH
        self.r_inner = r_inner
        self.lambda_ = np.array([])
        self.spec = np.array([])
        self.spec_0 = np.array([])
        self.Q_hat_0 = np.nan
        self.Q_0 = np.nan
        self.M_star = np.nan
        self.filling_factor = 1.0

    def set_stellar_spectrum(self):
        '''
        Set internal parameters lambda_ spec, spec_0
        :return: Nothing, update internal parameters
        '''
        self.lambda_, self.spec = self.stellar_spectrum.load_spectrum(self.age, self.logZ)
        __          , self.spec_0 = self.stellar_spectrum.load_spectrum(0.0, self.logZ)

    def calc_Q_hat(self, lambda_, spec):
        '''
        Function to calculate the number of ionizing photons
        :param lambda_: Wavelength values of the spectrum bins in AA
        :param spec: Luminosity values of spectrum bins in erg*s^-1*AA^-1
        :return: Rate of ionizing photons
        '''
        h = const.h.to(u.erg * u.s ** -1).value  # erg * s^-1
        c = const.c.to(cds.AA * u.s ** -1).value  # AA * s^-1

        mask = lambda_ <= 912.
        integrand = lambda_[mask] * spec[mask]
        integral = simps(lambda_[mask] * spec[mask], lambda_[mask])
        return 1 / (h * c) * integral

    def calc_Q_0(self):
        '''
        Calculate rate of ionizing photons necessary to satisfy conditions at zero-age.
        :return: Rate of ionizing photons
        '''
        c = const.c.to(u.cm * u.s**-1).value # cm * s^-1
        self.Q_0 = 10 ** self.logU * 4 * np.pi * self.r_inner ** 2 * self.nH * c
        return self.Q_0

    def calc_M_star(self):
        '''
        Calculate the stellar mass necessary to match Q_hat_0 and Q_0
        :return: Stellar mass in solar masses
        '''
        self.Q_hat_0 = self.calc_Q_hat(self.lambda_, self.spec_0)
        self.Q_0 = self.calc_Q_0()
        self.M_star = self.Q_0 / self.Q_hat_0
        return self.M_star

    def calc_Q(self):
        '''
        Calculate the rate of ionizing photons at the current age.
        :return: Rate of ionizing photons at current age.
        '''
        self.Q = self.M_star * self.calc_Q_hat(self.lambda_, self.spec)
        return self.Q

    def calc_all_parameters(self):
        '''
        Perform all calculations
        :return: Nothing, update internal parameters
        '''
        self.set_stellar_spectrum()
        self.calc_Q_0()
        self.calc_M_star()
        self.calc_Q()