import numpy as np
import pyCloudy as pc
from CloudyGalaxy.gas_stats import GasStatsBy17 as GasStats
from CloudyGalaxy.abundances import ElementalAbundances, convert_xi_to_F
from CloudyGalaxy.dust_attenuation import transmission_function
from CloudyGalaxy.read_write_cloudy import StellarSpectrum, format_emission_table, make_input_file, make_emission_line_files

n_proc              = 6

output_dir          = './output_data/'
model_name          = 'test_model'
data_path           = './input_data/'
star_table_filename = 'Chabrier_constant_SFH_IMF_lower_0p08_IMF_upper_120_star_table.ASCII'
emission_line_list  = format_emission_table('./input_data/cloudyLines_2021-03-29.dat')

nlogZs = 6
nlogUs = 2
nxis   = 2
ntaus  = 4

logZs = np.linspace(-1.0, 0.5, nlogZs)
logUs = np.linspace(-3.5, -1.5, nlogUs)
xis   = np.linspace(0.1, 0.6, nxis)
taus  = np.linspace(0.01, 4.0, ntaus)
Fs    = np.ones((nlogZs,nxis))*np.nan

pc.print_make_file(dir_ = output_dir)
for i in range(nlogZs):
    logZ = logZs[i]
    xi_to_F = convert_xi_to_F(logZ)
    for j in range(nlogUs):
        for k in range(nxis):
            logU = logUs[j]
            xi = xis[k]
            F = xi_to_F(xi)
            Fs[i,k] = F
            print(xi_to_F(xi))
            stellar_spectrum = StellarSpectrum(data_path, star_table_filename)
            gas_stats = GasStats(stellar_spectrum, logZ, logU)
            gas_stats.calc_all_parameters()
            elemental_abundances = ElementalAbundances(logZ, F)
            elemental_abundances.calc_all_parameters()
            make_input_file(output_dir, model_name, logZ, logU, xi, emission_line_list, stellar_spectrum, gas_stats, elemental_abundances)

pc.run_cloudy(dir_ = output_dir, n_proc = n_proc, model_name = model_name, use_make = True)

make_emission_line_files(output_dir, model_name, logZs, logUs, xis, taus, Fs)
