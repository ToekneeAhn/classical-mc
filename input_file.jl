
#------------------------------------------
# constants
#------------------------------------------
k_B = 1/11.6 # meV/K
mu_B = 0.67*k_B # K/T to meV/T

#------------------------------------------
# set MC parameters
#------------------------------------------
N_therm = Int(2*10^4) #thermalization sweeps
N_det = Int(10^4) #deterministic sweeps
N_meas = Int(2*10^4) #measurement sweeps
probe_rate = Int(100) #number of sweeps between measurements, larger reduces autocorrelation between samples
overrelax_rate = Int(10) #ratio of overrelax sweeps to metropolis sweeps
replica_exchange_rate = Int(50) #how many sweeps between replica exchanges

#------------------------------------------
# set lattice parameters 
#------------------------------------------
N = Int(4) #number of unit cells in one direction
S = 0.5

#------------------------------------------
# set Zeeman coupling parameters
#------------------------------------------
h_strength = 1.0
h = h_strength*[1,1,1]/sqrt(3) #magnetic field in global frame

#------------------------------------------
# set interaction parameters in K
#------------------------------------------

Js = [3.0, -0.5, 0.04, 0.0]

#------------------------------------------
# for parallel tempering: set min and max rank temperatures in units of K
#------------------------------------------
T_min = 0.25
T_max = 1.0