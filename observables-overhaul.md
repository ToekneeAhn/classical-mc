Make the `Observables` struct more general by: 
- [ ] Make `collect_hsweep` better, e.g. the logic in stacking the data is broken for vector data
- [ ] Related: add global magnetization to saved observables 
- [ ] Make the `collect_theta_sweep` better
- [ ] Allow user to define a list of observables that are tracked during simulation 
- [ ] Support other types of susceptibility (e.g. quadrupolar temperature and field susceptibility)? 
- [x] Define a `measure!` function which performs all the measurements, for use in parallel tempering and simulated annealing loops
- [x] Add global magnetization vector measurement
- [x] Redo `avg_spin` and make it a matrix of `ErrorPropagator`
- [x] Make a dedicated field to hold all the measured observables (e.g. susceptibility, specific heat) to make I/O easier (`write_hdf5.jl`)

