#local z axis on sublattice m in in column m+1
Z_LOCAL = 1/sqrt(3)*[1 1 1; 1 -1 -1; -1 1 -1; -1 -1 1]'

#local dipole moments. for non-kramers, only the z component is dipolar
LOCAL_INTERACTIONS = 1.0 .* [0 0 1; 0 0 1; 0 0 1; 0 0 1]'

#bond-dependent gamma factor
OMEGA = exp(2*pi*im/3)
GAMMA_IJ = [0 1 OMEGA OMEGA^2; 1 0 OMEGA^2 OMEGA; OMEGA OMEGA^2 0 1; OMEGA^2 OMEGA 1 0]

LOCAL_1 = [-2/sqrt(6) 1/sqrt(6) 1/sqrt(6); 0 -1/sqrt(2) 1/sqrt(2); 1/sqrt(3) 1/sqrt(3) 1/sqrt(3)]'
LOCAL_2 = [-2/sqrt(6) -1/sqrt(6) -1/sqrt(6); 0 1/sqrt(2) -1/sqrt(2); 1/sqrt(3) -1/sqrt(3) -1/sqrt(3)]'
LOCAL_3 = [2/sqrt(6) 1/sqrt(6) -1/sqrt(6); 0 -1/sqrt(2) -1/sqrt(2); -1/sqrt(3) 1/sqrt(3) -1/sqrt(3)]'
LOCAL_4 = [2/sqrt(6) -1/sqrt(6) 1/sqrt(6); 0 1/sqrt(2) 1/sqrt(2); -1/sqrt(3) -1/sqrt(3) 1/sqrt(3)]'
LOCAL_BASES = [Matrix{Float64}(LOCAL_1), Matrix{Float64}(LOCAL_2), Matrix{Float64}(LOCAL_3), Matrix{Float64}(LOCAL_4)]