Make the `Observables` struct more general by: 
- [ ] Better input of magnetic field direction for base parallel tempering and simulated annealing scripts?
- [ ] Allow user to define a list of observables that are tracked during simulation?
- [x] Save plane vectors as a group `plane_vectors[{plane_vector_1, plane_vector_2}]` for theta sweep jobs
- [x] We need to restore the `save_config` block in `params_theta_sweep.yaml` because `load_config` assumes it exists. 
- [x] Support other types of susceptibility (e.g. quadrupolar temperature and field susceptibility)? 
- [x] Make `collect_hsweep` better, e.g. the logic in stacking the data is broken for vector data
- [x] Related: add global magnetization to saved observables 
- [x] Define a `measure!` function which performs all the measurements, for use in parallel tempering and simulated annealing loops
- [x] Add global magnetization vector measurement
- [x] Redo `avg_spin` and make it a matrix of `ErrorPropagator`
- [x] Make a dedicated field to hold all the measured observables (e.g. susceptibility, specific heat) to make I/O easier (`write_hdf5.jl`)

