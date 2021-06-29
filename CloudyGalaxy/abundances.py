import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as InterpUS

class ElementalAbundances:
    '''
    Calculates the abundance of elements.
    '''
    def __init__(self, logZ, F):
        '''
        :param logZ: Log metallicity in units of solar metallicity
        :param F: depletion strength factor
        '''
        self.logZ = logZ
        self.F = F
        self.coef_A = dict(C=-0.60,    # coefficient A for depletion fractions as in Byler et al. 2017
                           N=-0.10,
                           O=-0.14,
                           Ne=-0.00,
                           Na=-2.00,
                           Mg=-2.16,
                           Al=-2.78,
                           Si=-1.62,
                           S=-0.00,
                           Cl=-2.00,
                           Ar=-0.00,
                           Ca=-5.04,
                           Fe=-2.62,
                           Ni=-4.00,
                           F=0.0,
                           P=0.0,
                           K=0.0,
                           Cr=0.0,
                           Ti=0.0,
                           Mn=0.0,
                           Co=0.0,
                           Cu=0.0,
                           Zn=0.0
                           )
        self.coef_B = dict(C=-0.30,    #coefficient B for depletion fractions as in Byler et al. 2017
                           N=-0.05,
                           O=-0.07,
                           Ne=-0.00,
                           Na=-1.00,
                           Mg=-1.08,
                           Al=-1.39,
                           Si=-0.81,
                           S=-0.00,
                           Cl=-1.00,
                           Ar=-0.00,
                           Ca=-2.52,
                           Fe=-1.31,
                           Ni=-2.00,
                           F=0.0,
                           P=0.0,
                           K=0.0,
                           Cr=0.0,
                           Ti=0.0,
                           Mn=0.0,
                           Co=0.0,
                           Cu=0.0,
                           Zn=0.0
                           )
        self.coef_z = dict(C=0.5,    #coefficient z for depletion fractions as in Byler et al. 2017
                           N=0.5,
                           O=0.5,
                           Ne=0.5,
                           Na=0.5,
                           Mg=0.5,
                           Al=0.5,
                           Si=0.5,
                           S=0.5,
                           Cl=0.5,
                           Ar=0.5,
                           Ca=0.5,
                           Fe=0.5,
                           Ni=0.5,
                           F=0.5,
                           P=0.5,
                           K=0.5,
                           Cr=0.5,
                           Ti=0.5,
                           Mn=0.5,
                           Co=0.5,
                           Cu=0.5,
                           Zn=0.5
                           )
        self.solar = dict(He=-1.01,
                          C=-3.57,
                          N=-4.60,
                          O=-3.31,
                          Ne=-4.07,
                          Na=-5.75,
                          Mg=-4.40,
                          Al=-5.55,
                          Si=-4.49,
                          S=-4.86,
                          Cl=-6.63,
                          Ar=-5.60,
                          Ca=-5.66,
                          Fe=-4.50,
                          Ni=-5.78,
                          F=-7.44,
                          P=-6.59,
                          K=-6.97,
                          Cr=-6.36,
                          Ti=-7.05,
                          Mn=-6.57,
                          Co=-7.01,
                          Cu=-7.81,
                          Zn=-7.44
                          )
        self.mass = dict(H=1.0079,    #Mass of elements as from https://www.lenntech.com/periodic/mass/atomic-mass.htm
                         He=4.0026,
                         C=12.011,
                         N=14.007,
                         O=15.999,
                         Ne=20.1797,
                         Na=22.9897,
                         Mg=24.305,
                         Al=26.9815,
                         Si=28.085,
                         S=32.06,
                         Cl=35.453,
                         Ar=39.948,
                         Ca=40.078,
                         Fe=55.845,
                         Ni=58.6934,
                         F=18.9984,
                         P=30.9738,
                         K=39.0983,
                         Cr=51.9961,
                         Ti=47.867,
                         Mn=54.938,
                         Co=58.9332,
                         Cu=63.546,
                         Zn=65.39
                         )

    def set_depletion(self):
        '''
        Set the depletion factor for all elements as a function of F
        :return: Nothing, updates internal depl parameter
        '''
        depl = dict.fromkeys(self.solar.keys())
        for key in self.coef_A.keys():
            depl[key] = self.coef_B[key] + self.coef_A[key] * (self.F - self.coef_z[key])
        depl['He'] = -0.00  # Dopita et al. 2013
        self.depl = depl

    def set_undepleted_abundance(self):
        '''
        Calculate the undepleted abundance of elements as a function of logZ
        :return: Nothing, updates internal undepl_abund parameter
        '''
        def calc_He(logZ):
            return np.log10(0.0737 + (0.024 * (10.0 ** self.logZ)))

        def calc_CNO(logZ):
            # Scaling for Nitrogen and Carbon as in Dopita et al. 2013
            oxy = np.array([7.39, 7.50, 7.69, 7.99, 8.17,
                            8.39, 8.69, 8.80, 8.99, 9.17, 9.39])
            nit = np.array([-6.61, -6.47, -6.23, -5.79, -5.51,
                            -5.14, -4.60, -4.40, -4.04, -3.67, -3.17])
            car = np.array([-5.58, -5.44, -5.20, -4.76, -4.48,
                            -4.11, -3.57, -3.37, -3.01, -2.64, -2.14])
            O = self.solar['O'] + self.logZ
            C = float(InterpUS(oxy, car, k=1)(O + 12.0))
            N = float(InterpUS(oxy, nit, k=1)(O + 12.0))
            return C, N, O

        undepl_abund = dict.fromkeys(self.solar.keys())
        undepl_abund['He'] = calc_He(self.logZ)
        undepl_abund['C'] = calc_CNO(self.logZ)[0]
        undepl_abund['N'] = calc_CNO(self.logZ)[1]
        undepl_abund['O'] = calc_CNO(self.logZ)[2]
        for key in undepl_abund.keys():
            if key != 'He' and key != 'C' and key != 'N' and key != 'O':
                undepl_abund[key] = self.solar[key] + self.logZ
        self.undepl_abund = undepl_abund
        return

    def set_abundance(self):
        '''
        Set the final abundance. Applying the depletion fraction on the undepleted abundance.
        :return: Nothing, updates internal abund parameter.
        '''
        abund = dict.fromkeys(self.solar.keys())
        for key in abund.keys():
            abund[key] = self.undepl_abund[key] + self.depl[key]
        self.abund = abund

    def calc_dust_to_metal(self):
        '''
        Calculates the dust-to-metal ('aka dust-to-heavy element') ratio (xi).
        :return: Nothing, updates internal dust_to_metal parameter
        '''
        numerator = 0.
        denominator = 0.
        for key in self.depl:
            if key != 'He' and key != 'H':
                numerator += self.mass[key] * (1 - 10 ** self.depl[key]) * 10 ** (self.undepl_abund[key])
                denominator += self.mass[key] * 10 ** (self.undepl_abund[key])
        self.dust_to_metal = numerator / denominator

    def calc_all_parameters(self):
        '''
        Calculate all internal parameters.
        :return: Nothing, updates internal parameters.
        '''
        self.set_depletion()
        self.set_undepleted_abundance()
        self.set_abundance()
        self.calc_dust_to_metal()

def convert_xi_to_F(logZ):
    '''
    Function to convert xi (dust-to-metal ratio) values to F to be inputted.
    :param logZ: Log metallicity as a function of solar metallicity.
    :return: Interpolated function to convert xi to F for the inputted metallicity.
    '''
    Fs = np.linspace(0.0, 4.0, 100)
    xis = np.zeros((100))
    for i in range(len(Fs)):
        abundances = ElementalAbundances(logZ,Fs[i])
        abundances.calc_all_parameters()
        xis[i] = abundances.dust_to_metal
    return InterpUS(xis, Fs, k=1)