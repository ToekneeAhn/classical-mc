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

# quadrupolar order parameter definitions for Kramers moments, see Patri (2020) table II first column
# Q_1 and Q_2 correspond to the E_g irrep of T_d (psi_2/psi_3)
Q_1_MASK = 0.5*[1 1 1 1;
                0 0 0 0;
                0 0 0 0]

Q_2_MASK = 0.5*[0 0 0 0;
                1 1 1 1;
                0 0 0 0]

#Q_xy, Q_xz, and Q_yz correspond to T_2g irrep of T_d (palmer-chalker for Kramers moments)
Q_yz_MASK = 0.5*[1 1 -1 -1;
                0 0 0 0;
                0 0 0 0]

Q_xz_MASK = 0.25*[-1 1 -1 1;
             sqrt(3) -sqrt(3) sqrt(3) -sqrt(3);
             0 0 0 0]

Q_xy_MASK = 0.25*[-1 1 1 -1;
             -sqrt(3) sqrt(3) sqrt(3) -sqrt(3);
             0 0 0 0]
             
#Q_x, Q_y, and Q_z correspond to T_1g irrep of T_d (splayed ferromagnet for Kramers moments)
Q_x_MASK = 0.5*[0 0 0 0;
                1 1 -1 -1;
                0 0 0 0]

Q_y_MASK = 0.25*[-sqrt(3) sqrt(3) -sqrt(3) sqrt(3);
                -1 1 -1 1;
                0 0 0 0]

Q_z_MASK = 0.25*[sqrt(3) -sqrt(3) -sqrt(3) sqrt(3);
                -1 1 1 -1;
                0 0 0 0]

QUADRUPOLAR_MASKS = Dict("Q1"=>Q_1_MASK, "Q2"=>Q_2_MASK, 
                        "Qxy"=>Q_xy_MASK, "Qxz"=>Q_xz_MASK, "Qyz"=>Q_yz_MASK, 
                        "Qx"=>Q_x_MASK, "Qy"=>Q_y_MASK, "Qz"=>Q_z_MASK)

# quadrupolar order parameters derived from these mask groupings
Q_SPECS = (
    (name = "Q_E", masks = ("Q1", "Q2")),
    (name = "Q_T1", masks = ("Qx", "Qy", "Qz")),
    (name = "Q_T2", masks = ("Qxy", "Qxz", "Qyz")),
)

# masks for the observable"spin_variance" 
# the variances of these operators appear in the correction to the elastic constants
# the magnetoelastic constants in front are assumed to be the same for simplicity
# the last row (z components) needs to be "dotted" with (hz_0, hz_1, hz_2, hz_3)
V_B_MASK = [0 0 0 0;
            0 0 0 0;
            1 1 1 1] 

# g_E = k1/sqrt(3) - k2/sqrt(6)
# note: z row is dotted with hy for mu and hx for nu
V_MU_MASK = [1 1 1 1;
            0 0 0 0;
            1 1 1 1]

V_NU_MASK = [0 0 0 0;
            1 1 1 1;
            1 1 1 1]

# a = (4 k1 + sqrt(2) k2)/3 for row x
# b = a*sqrt(3)/2 for row y
# 2/3*(g3-g4) for row z
V_XY_MASK = [1 -1 -1 1;
             0 0 0 0;
             -1 1 1 -1]

V_XZ_MASK = [-1/2 1/2 -1/2 1/2;
             -sqrt(3)/2 -sqrt(3)/2 sqrt(3)/2 sqrt(3)/2;
             -1 1 -1 1]

V_YZ_MASK = [-1/2 -1/2 1/2 1/2;
             sqrt(3)/2 sqrt(3)/2 -sqrt(3)/2 -sqrt(3)/2;
             -1 -1 1 1]

ELASTIC_CORRECTION_SPECS = (
    (name = "V_B", mask = V_B_MASK),
    (name = "V_MU", mask = V_MU_MASK),
    (name = "V_NU", mask = V_NU_MASK),
    (name = "V_XY", mask = V_XY_MASK),
    (name = "V_XZ", mask = V_XZ_MASK),
    (name = "V_YZ", mask = V_YZ_MASK)
)

# I/O 
PARAMETER_FIELDS = ["N_therm", "N_meas", "overrelax_rate", "probe_rate", "replica_exchange_rate",
                    "N", "S", "Js", "K", "delta_12", "Ts", 
                    "h_direction", "h_sweep", "disorder_strength", "disorder_seed"]

OBSERVABLE_FIELDS = ["magnetization", "magnetization_global", "energy", "specific_heat", "susceptibility", 
                    "binder_cumulant", "local_spin", "dSdT", "dQdT", "Q",
                    "magnetization_err", "magnetization_global_err", "energy_err", "specific_heat_err", "susceptibility_err", 
                    "binder_cumulant_err", "local_spin_err", "dSdT_err", "dQdT_err", "Q_err", "elastic_correction", "elastic_correction_err",
                    "spins"]
