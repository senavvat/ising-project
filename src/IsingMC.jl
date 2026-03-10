module IsingMC

"""IsingMC — simple 2D Ising Monte Carlo package

This module re-uses the implementation in `src/init_spins.jl` and
exposes a small public API for simulations.
"""

export IsingState, init_state, metropolis_step!, metropolis_sweep!, magnetization, total_energy, delta_energy_flip, flip!

include("init_spins.jl")

end # module
