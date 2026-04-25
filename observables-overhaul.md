Make the `Observables` struct more general by: 
- [ ] Allowing user to define a list of observables that are tracked during simulation 
- [ ] Supporting other types of susceptibility (e.g. quadrupolar temperature and field susceptibility)
- [x] Define a `measure!` function which performs all the measurements, for use in parallel tempering and simulated annealing loops
- [x] Add global magnetization vector measurement
- [x] Redo `avg_spin` and make it a matrix of `ErrorPropagator`
- [x] Make a dedicated field to hold all the measured observables (e.g. susceptibility, specific heat) to make I/O easier (`write_hdf5.jl`)
- [ ] Make the collection functions cleaner
