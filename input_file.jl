
#------------------------------------------
# constants
#------------------------------------------
k_B = 1/11.6 # meV/K
mu_B = 0.67*k_B # K/T to meV/T

#------------------------------------------
# local z axis for each basis site 
#------------------------------------------
#=
z1 = [1, 1, 1]/sqrt(3)
z2 = [1,-1,-1]/sqrt(3)
z3 = [-1,1,-1]/sqrt(3)
z4 = [-1,-1,1]/sqrt(3)
=#

#------------------------------------------
# set MC parameters
#------------------------------------------

N_therm = Int(10^5) #thermalization sweeps
N_det = Int(10^4) #deterministic sweeps
N_meas = Int(10^4) #measurement sweeps
probe_rate = Int(100) #number of sweeps between measurements, larger reduces autocorrelation between samples
overrelax_rate = Int(10) #ratio of overrelax sweeps to metropolis sweeps
replica_exchange_rate = Int(50) #how many sweeps between replica exchanges

#------------------------------------------
# set lattice parameters 
#------------------------------------------
N = Int(6) #number of unit cells in one direction
S = 1/2


#------------------------------------------
# set Zeeman coupling parameters
#------------------------------------------
#for single runs
h_strength = 1.5
h = h_strength*[1,1,1]/sqrt(3) #magnetic field in global frame

#h sweep where the sweep values are defined in input_file.jl
N_h = 20
h_sweep = range(0.0, 12.0, N_h)

#------------------------------------------
# set interaction parameters in meV
#------------------------------------------
Js = [1.0, 0.02, 0.05, 0.0] #J_zz, J_pm, J_pmpm, J_zpm
#Js = (1.0, 0.5, 0.05, 0.0) #J_zz, J_pm, J_pmpm, J_zpm

#------------------------------------------
# for parallel tempering: set min and max rank temperatures in units of meV
#------------------------------------------
T_min = 0.01*k_B
T_max = 1.0*k_B