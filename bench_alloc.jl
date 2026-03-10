include("src/init_spins.jl")
state = init_state(50, seed=123)

# measure shuffle! allocations
a = Base.@allocated(shuffle!(state.rng, state.idx))
println("alloc shuffle=", a)

# measure loop allocations
b = Base.@allocated(begin
    for k in state.idx
        i = div(k-1, 50) + 1
        j = (k-1) % 50 + 1
        metropolis_trial!(state, i, j)
    end
end)
println("alloc loop=", b)

# measure entire sweep allocations
c = Base.@allocated(metropolis_sweep!(state))
println("alloc sweep=", c)
