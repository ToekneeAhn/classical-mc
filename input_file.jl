
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

#=
probe_rate=2000
report_interval = Int(1e2)
checkpoint_rate=100
=#
N_therm = 10^3 #thermalization sweeps
N_det = 10^4 #deterministic sweeps, although not really for pt
probe_rate = 100 #number of sweeps between measurements, larger reduces autocorrelation between samples
overrelax_rate = 10 #ratio of overrelax sweeps to metropolis sweeps
replica_exchange_rate = 50 #how many sweeps between replica exchanges

#------------------------------------------
# set lattice parameters 
#------------------------------------------
N = 2 #number of unit cells in one direction
S = 1/2

#------------------------------------------
# set Zeeman coupling parameters
#------------------------------------------
#=
gxx = 0.0
gzz = 2.18
gyy = 0.0
h = [1, 1, 1]/sqrt(3)
h1 = (h'*z1) * [gxx, gyy, gzz]
h2 = (h'*z2) * [gxx, gyy, gzz]
h3 = (h'*z3) * [gxx, gyy, gzz]
h4 = (h'*z4) * [gxx, gyy, gzz]
=#

#for single runs
#h_strength = 2.96
h_strength=0.0
h = h_strength*[1,1,1]/sqrt(3) #magnetic field in global frame

#h sweep
#N_h = 15
#h_sweep = range(2.7, 3.3, N_h)

#------------------------------------------
# set interaction parameters in meV
#------------------------------------------
Js = (1.0, 0.02, 0.05, 0.0) #J_zz, J_pm, J_pmpm, J_zpm

#------------------------------------------
# set min and max rank temperatures in units of meV
#------------------------------------------
T_min = 0.01*k_B
T_max = 1.0*k_B