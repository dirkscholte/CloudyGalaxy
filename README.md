# CloudyGalaxy

Code to run Cloudy 17.02 for conditions appropriate to galaxy H2 regions. 

- Uses FSPS irradiating spectra. (imported through StellarSpectrum class in read_write_cloudy.py)
- Abundances as in Asplund et al. (2009) and Grevesse et al. (2010) (in abundances.py)
- Variable depletion factors. Log(D) scaled linearly, anchored at log(D) is 0 and log(D) as in Dopita et al. (2013) (in abundances.py)
- Parametrization of the geometry and gas distribution in two ways: the parametrization as in Byler et al. (2017) and the parametrization as in Charlot and Longhetti (2001). (in gas_stats.py)
- Creates input files for Cloudy 17.02 using pyCloudy. (in read_write_cloudy.py)
- Dust attenuation through the Charlot and Fall (2000) models. (in dust_attenuation.py)
- Emission lines extracted from Cloudy files and collected in .npy files. (in read_write_cloudy.py)
