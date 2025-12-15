
#------------------------------------------
# constants
#------------------------------------------
k_B = 1/11.6 # meV/K
mu_B = 0.67*k_B # K/T to meV/T

#------------------------------------------
# set MC parameters
#------------------------------------------
N_therm = Int(10^3) #thermalization sweeps
N_det = Int(10^4) #deterministic sweeps
N_meas = Int(10^4) #measurement sweeps
probe_rate = Int(100) #number of sweeps between measurements, larger reduces autocorrelation between samples
overrelax_rate = Int(10) #ratio of overrelax sweeps to metropolis sweeps
replica_exchange_rate = Int(50) #how many sweeps between replica exchanges
optimize_temperature_rate = Int(10^6)

#------------------------------------------
# set lattice parameters 
#------------------------------------------
N = Int(4) #number of unit cells in one direction
S = 0.5

#------------------------------------------
# set Zeeman coupling parameters
#------------------------------------------
h_strength = 7.0
h_direction = [1,1,1]/sqrt(3)
h = h_strength*h_direction #magnetic field in global frame

delta_12 = [0.0, 0.0] #quadratic Zeeman field strengths
disorder_strength = 0.0 #Gamma parameter in Lorentzian distribution
#------------------------------------------
# set interaction parameters in K
#------------------------------------------

Js = [1.0, 0.02, 0.05, 0.0]

include_cubic = true #whether to include cubic interaction term
K = 0.1 + 0.1im #complex scalar for cubic interaction strength

#------------------------------------------
# for simulated annealing
#------------------------------------------

T_i = 1.0 #initial temperatures
T_f = 1e-4 #target temperature

#------------------------------------------
# for parallel tempering: set min and max rank temperatures in units of K
#------------------------------------------
T_min = 0.33
T_max = 1.0