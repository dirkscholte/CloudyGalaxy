import math
import numpy as np
import pyCloudy as pc
from dust_attenuation import transmission_function
from surface_density import calc_log_dust, calc_log_gas

class StellarSpectrum:
    '''
    Specifies location of stellar spectrum (SSP) files, formats input text for Cloudy and loads spectra.
    '''
    def __init__(self, data_path, filename):
        '''
        :param data_path: Path to stellar spectrum (SSP) files
        :param filename: Filename of stellar spectrum file
        '''
        self.data_path = data_path
        self.filename = filename

    def star_table_string(self, logZ, age=1e8):
        '''
        Create string of text to be pasted into the Cloudy input file.
        :param logZ: Log metallicity in units of solar metallicity.
        :param age: Age of the SSP in years
        :return: String to be pasted in Cloudy input file.
        '''
        return 'table star "{0}" age={1:.2e} logz={2:.2e}'.format(self.filename, age, logZ)

    def load_spectrum(self, age, logZ, f_lambda_in=True):
        '''
        Loading of spectra from file and writing them to numpy arrays
        :param age: Age of SSP in years
        :param logZ: Log metallicity in units of solar metallicity
        :param f_lambda_in: If True: Luminosity on file in units of wavelength (erg*s^-1*AA^-1). Else: Luminosity on file in units of frequency (erg*s^-1*Hz^-1).
        :return: Two numpy arrays of equal length. One with the wavelengths of the spectrum bins and one with the Luminosities.
        '''
        f = open(self.data_path + self.filename, "r")
        lines = f.readlines()
        header_lines = 11
        __, ndims, npars, par1, par2, nmod, nx, x, conv1, f_type, conv2 = lines[0:header_lines]
        nmod_lines = math.ceil(int(nmod) / 4)
        nx_lines = math.ceil(int(nx) / 5)
        conv1 = float(conv1)
        conv2 = float(conv2)

        mods = lines[11:nmod_lines + 11]
        mods = list(map(str.split, mods))
        mods = [item for sublist in mods for item in sublist]
        mods = [float(item) for item in mods]
        mods = np.array(mods).reshape(-1, 2)

        mod_no = np.argmin(np.sum(np.abs(mods - np.array([age, logZ])), axis=-1))

        x_vals = lines[nmod_lines + 11:nx_lines + nmod_lines + 11]
        x_vals = list(map(str.split, x_vals))
        x_vals = [item for sublist in x_vals for item in sublist]
        x_vals = [float(item) for item in x_vals]
        x_vals = np.array(x_vals)

        y_start = mod_no * nx_lines + nx_lines + nmod_lines + 11
        y_stop = (mod_no + 1) * nx_lines + nx_lines + nmod_lines + 11

        y_vals = lines[y_start:y_stop]
        y_vals = list(map(str.split, y_vals))
        y_vals = [item for sublist in y_vals for item in sublist]
        y_vals = [float(item) for item in y_vals]
        y_vals = np.array(y_vals)

        f.close()

        if f_lambda_in:
            lambda_ = x_vals
            f_lambda = y_vals * conv2
            return lambda_, f_lambda
        else:
            lambda_ = x_vals
            f_nu = y_vals * conv2
            c = 2.9979e18  # ang s^-1
            f_lambda = f_nu * c / lambda_ ** 2  # erg s^-1 AA^-1
            return lambda_, f_lambda

def format_emission_table(full_filename):
    '''
    Formats the emission table file to be inputted in PyCloudy
    :param full_filename: Path + filename of the list of emission lines.
    :return: list of emission lines in correct format.
    '''
    data = open(full_filename,'r').readlines()
    stripped_data = [line.strip('\n').strip('\t') for line in data]
    return stripped_data


def make_input_file(output_dir, model_name, logZ, logU, xi, emission_line_list, stellar_spectrum, gas_stats, elemental_abundances, age=1e6):
    '''
    Writes the Cloudy input file
    :param output_dir: Directory to write Cloudy input file
    :param model_name: Name of model
    :param logZ: Log metallicity in units of solar metallicity
    :param logU: Log ionization parameter
    :param xi: Dust-to-metal ratio
    :param emission_line_list: List of emission lines to be tracked and saved to file by Cloudy
    :param stellar_spectrum: Instance of the Stellar Spectrum class with data paths defined.
    :param gas_stats: Instance of the GasStats class.
    :param elemental_abundances: Instance of the ElementalAbundances class.
    :param age: Age of the SSP in years
    :return: Nothing, writes to output_dir+model_name+'.in'
    '''
    full_model_name = '{0}_{1:.3e}_{2:.3e}_{3:.3e}'.format(model_name, logZ, logU, xi)
    c_input = pc.CloudyInput('{0}{1}'.format(output_dir, full_model_name)) # create input object for cloudy model

    c_input.set_star(SED=stellar_spectrum.star_table_string(logZ, age=age),
                     SED_params='',
                     lumi_unit='Q(H)',
                     lumi_value= np.log10(gas_stats.Q)
                    )
    c_input.set_radius(np.log10(gas_stats.r_inner)) # Set inner radius of gas cloud
    c_input.set_abund(ab_dict= elemental_abundances.abund,
                      nograins = True
                     ) # Set abundances
    c_input.set_cste_density(dens=np.log10(gas_stats.nH)) # set constant hydrogen density
    c_input.set_emis_tab(emission_line_list) # set emission line table
    c_input.set_sphere(True) #Spherical geometry
    c_input.set_iterate(to_convergence=True) # iterate to convergence
    c_input.set_stop(['temperature 100.0', 'efrac -1.00']) # stop criteria
    options = ('print line precision 6', 'COSMIC RAY BACKGROUND')
    c_input.set_other(options)

    c_input.print_input(to_file = True, verbose = False)

def make_emission_line_files(output_dir, model_name, logZs, logUs, xis, taus, Fs):
    '''
    Reads Cloudy output files and extracts emission line fluxes. Writes to files along with line labels, line wavelengths and parameter values.
    :param output_dir: Output directory of the data files.
    :param model_name: Name of the model
    :param logZs: Log metallicity in units of solar metallicity.
    :param logUs: Log ionization parameter.
    :param xis: Dust-to-metal ratio
    :param taus: Optical depth at 5500 Angstrom
    :param calc_F: Function to calculate the depletion strength factor
    :return: 4 '.npy' files containing the emission line labels, wavelengths, parameter settings, line luminosity.
    '''
    nlogZs = len(logZs)
    nlogUs = len(logUs)
    nxis   = len(xis)
    ntaus  = len(taus)

    emline_param_cube = np.zeros((nlogZs,nlogUs,nxis,ntaus, 7))
    emline_luminosity_cube = np.zeros((nlogZs,nlogUs,nxis,ntaus, 127))

    for i in range(nlogZs):
        for j in range(nlogUs):
            for k in range(nxis):
                logZ = logZs[i]
                logU = logUs[j]
                xi   = xis[k]
                F    = Fs[i,k]
                full_model_name = '{0}_{1:.3e}_{2:.3e}_{3:.3e}'.format(model_name, logZ, logU, xi)
                print('{0}{1}'.format(output_dir,full_model_name))
                models = pc.load_models('{0}{1}'.format(output_dir,full_model_name), read_grains = False)
                print('{} models found!'.format(len(models)))
                model = models[0]
                for l in range(ntaus):
                    tau  = taus[l]
                    print('PROCESSING: logZ = {0:06.4f}, logU = {1:06.4f}, xi = {2:06.4f}, tau = {3:06.4f}.'.format(logZ, logU, xi, tau))

                    lambda_emission_lines = [float(label[5:-1])*0.01 for label in model.emis_labels] #Ang
                    emission_lines = [model.get_emis_vol(ref=label) for label in model.emis_labels] #erg/s
                    attenuated_emission_lines = emission_lines * transmission_function(lambda_emission_lines, tau)
                    emline_param_cube[i,j,k,l] = np.array([logZ, logU, xi, tau, F, calc_log_dust(tau), calc_log_gas(logZ, xi, tau)])
                    emline_luminosity_cube[i,j,k,l] = attenuated_emission_lines
                    if i==0 and j==0 and k==0 and l==0:
                        emline_labels = model.emis_labels
                        emline_lambda = [float(label[5:-1])*0.01 for label in model.emis_labels] #Ang

    np.save(model_name + '_emission_line_labels.npy', emline_labels)
    np.save(model_name + '_emission_line_wavelengths.npy', emline_lambda)
    np.save(model_name + '_parameters_file.npy', emline_param_cube)
    np.save(model_name + '_emission_line_luminosity_file.npy', emline_luminosity_cube)
