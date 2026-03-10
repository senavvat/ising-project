using IsingMC
using Statistics

"""
run_mc_stats(L, T; n_therm=50, n_steps=200, seed=123)

短いモンテカルロを実行して E, M, M^2 の平均を返す。
返り値は Dict(:T=>T, :L=>L, :E=>meanE, :M=>meanM, :M2=>meanM2)
"""
function run_mc_stats(L::Integer, T::Real; n_therm::Integer=50, n_steps::Integer=200, seed=nothing)
    beta = 1.0 / float(T)
    s = init_state(L, J=1.0, beta=beta, seed=seed)

    # 熱平衡化
    for _ in 1:n_therm
        metropolis_sweep!(s)
    end

    Es = Float64[]
    Ms = Float64[]
    for _ in 1:n_steps
        metropolis_sweep!(s)
        push!(Es, total_energy(s))
        push!(Ms, magnetization(s))
    end

    meanE = mean(Es)
    meanM = mean(Ms)
    meanM2 = mean(x->x^2, Ms)

    return Dict(:T=>T, :L=>L, :E=>meanE, :M=>meanM, :M2=>meanM2)
end

function scan_and_plot(; L=16, temps=collect(0.5:0.5:3.5), n_therm=50, n_steps=200, seed=123, savepath="analysis/m2_vs_T.png")
    results = Dict{Float64,Dict}()
    Ts = Float64[]
    M2s = Float64[]
    println("Running temperature scan: L=$(L), n_steps=$(n_steps)")
    for T in temps
        stats = run_mc_stats(L, T; n_therm=n_therm, n_steps=n_steps, seed=seed)
        results[T] = stats
        push!(Ts, T)
        push!(M2s, stats[:M2])
        @info "T=$T => M2=$(round(stats[:M2], digits=4))"
    end

    # プロット：Plots.jl を使う（無ければ実行手順を案内）
    try
        @eval using Plots
    catch err
        println("Plots.jl is not available in the current environment.")
        println("Install it and re-run, e.g.:")
        println("  julia --project=. -e 'using Pkg; Pkg.add(\"Plots\")'")
        return results
    end

    plt = plot(Ts, M2s, marker=:circle, xlabel="T", ylabel="<M^2>", title="<M^2> vs T (L=$(L))")
    savefig(plt, savepath)
    println("Saved plot to: $savepath")
    return results
end

if abspath(PROGRAM_FILE) == @__FILE__
    scan_and_plot()
end
