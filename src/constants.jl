#local z axis on sublattice m in in column m+1
Z_LOCAL = 1/sqrt(3)*[1 1 1; 
                    1 -1 -1; 
                    -1 1 -1; 
                    -1 -1 1]'

#local dipole moments. for non-kramers, only the z component is dipolar
LOCAL_INTERACTIONS = 1.0 .* [0 0 1; 
                            0 0 1; 
                            0 0 1; 
                            0 0 1]'

#bond-dependent gamma factor
OMEGA = exp(2*pi*im/3)
GAMMA_IJ = [0 1 OMEGA OMEGA^2; 
            1 0 OMEGA^2 OMEGA; 
            OMEGA OMEGA^2 0 1; 
            OMEGA^2 OMEGA 1 0]

LOCAL_1 = [-2/sqrt(6) 1/sqrt(6) 1/sqrt(6); 0 -1/sqrt(2) 1/sqrt(2); 1/sqrt(3) 1/sqrt(3) 1/sqrt(3)]'
LOCAL_2 = [-2/sqrt(6) -1/sqrt(6) -1/sqrt(6); 0 1/sqrt(2) -1/sqrt(2); 1/sqrt(3) -1/sqrt(3) -1/sqrt(3)]'
LOCAL_3 = [2/sqrt(6) 1/sqrt(6) -1/sqrt(6); 0 -1/sqrt(2) -1/sqrt(2); -1/sqrt(3) 1/sqrt(3) -1/sqrt(3)]'
LOCAL_4 = [2/sqrt(6) -1/sqrt(6) 1/sqrt(6); 0 1/sqrt(2) 1/sqrt(2); -1/sqrt(3) -1/sqrt(3) 1/sqrt(3)]'
LOCAL_BASES = [Matrix{Float64}(LOCAL_1), Matrix{Float64}(LOCAL_2), Matrix{Float64}(LOCAL_3), Matrix{Float64}(LOCAL_4)]

# quadrupolar order parameter definitions
Q_1_MASK = [sqrt(3) sqrt(3) sqrt(3) sqrt(3);
            -1 -1 -1 -1;
            0 0 0 0]

Q_2_MASK = [1 1 1 1;
            sqrt(3) sqrt(3) sqrt(3) sqrt(3);
            0 0 0 0]

Q_xy_MASK = [-1 1 1 -1;
             -sqrt(3) sqrt(3) sqrt(3) -sqrt(3)
             0 0 0 0]

Q_xz_MASK = [-1 1 -1 1;
             sqrt(3) -sqrt(3) sqrt(3) -sqrt(3)
             0 0 0 0]

Q_yz_MASK = [1 1 -1 -1;
             0 0 0 0;
             0 0 0 0] 

# order in which the quadrupolar susceptibilities are calculated and output
Q_MASKS = Dict(1=>Q_1_MASK, 2=>Q_2_MASK, 3=>Q_xy_MASK, 4=>Q_xz_MASK, 5=>Q_yz_MASK)

# I/O 
PARAMETER_FIELDS = ["N_therm", "N_meas", "overrelax_rate", "probe_rate", "replica_exchange_rate",
                    "N", "S", "Js", "K", "delta_12", "Ts", 
                    "h_direction", "h_sweep", "disorder_strength", "disorder_seed"]

OBSERVABLE_FIELDS = ["magnetization", "magnetization_global", "energy", "specific_heat", "susceptibility", 
                    "binder_cumulant", "local_spin", "dSdT", "dQdT",
                    "magnetization_err", "magnetization_global_err", "energy_err", "specific_heat_err", "susceptibility_err", 
                    "binder_cumulant_err", "local_spin_err", "dSdT_err", "dQdT_err",
                    "spins"]
