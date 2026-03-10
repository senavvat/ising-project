using Test
using IsingMC

@testset "IsingMC basic tests" begin
    @testset "init_state" begin
        s = init_state(8, seed=1)
        @test size(s.spins) == (8, 8)
        @test all(x -> x == 1 || x == -1, vec(s.spins))
    end

    @testset "energy consistency" begin
        s = init_state(4, seed=2)
        e0 = total_energy(s)
        de = delta_energy_flip(s, 1, 1)
        flip!(s, 1, 1)
        e1 = total_energy(s)
        @test isapprox(e1 - e0, de)
    end
end
