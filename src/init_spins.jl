using Random

# Minimal container for simulation state
mutable struct IsingState{T<:Integer}
    spins::Array{T,2}
    J::Float64
    beta::Float64
    rng::AbstractRNG
    idx::Vector{Int}
    i_idx::Vector{Int}
    j_idx::Vector{Int}
end

"""
init_state(L; J=1.0, beta=1.0, eltype=Int8, seed=nothing)

便利なコンストラクタ: 格子を初期化して `IsingState` を返す。
"""
function init_state(L::Integer; J=1.0, beta=1.0, eltype::Type=Int8, seed=nothing)
    rng = seed === nothing ? Random.GLOBAL_RNG : MersenneTwister(seed)
    spins = init_spins(L; eltype=eltype, seed=seed)
    N = L * L
    idx = collect(1:N)
    i_idx = Vector{Int}(undef, N)
    j_idx = Vector{Int}(undef, N)
    for k in 1:N
        i_idx[k] = div(k-1, L) + 1
        j_idx[k] = (k-1) % L + 1
    end
    return IsingState{eltype}(spins, float(J), float(beta), rng, idx, i_idx, j_idx)
end

"""
init_spins(L; eltype=Int8, seed=nothing)

L x L のスピン格子を返す。各要素は +1 または -1 の整数型。
- `eltype` : 要素型（デフォルト `Int8`）
- `seed`   : 再現用の乱数シード（省略時はランダム）
"""
function init_spins(L::Integer; eltype::Type=Int8, seed=nothing)
    L = Int(L)
    @assert L > 0 "L must be positive"

    rng = seed === nothing ? Random.GLOBAL_RNG : MersenneTwister(seed)
    spins = rand(rng, (-1, 1), L, L)
    return eltype.(spins)
end


"""
sum_neighbors(spins, i, j)

与えられた格子 `spins` に対して、サイト `(i,j)` の4近傍スピンの和を返す。
周期境界条件 (periodic boundary conditions) を採用しており、格子端の隣接は反対側に回り込む。
インデックスは 1-based を想定する。
"""
function sum_neighbors(spins::AbstractMatrix, i::Integer, j::Integer)
    L1, L2 = size(spins)
    @assert L1 == L2 "square lattice expected"
    L = L1
    up    = spins[mod1(i-1, L), j]
    down  = spins[mod1(i+1, L), j]
    left  = spins[i, mod1(j-1, L)]
    right = spins[i, mod1(j+1, L)]
    return up + down + left + right
end


"""
local_energy(spins, i, j; J=1)

サイト `(i,j)` に関する局所エネルギーを計算する（結合定数 `J` を指定可能）。
定義: E_i = -J * s_i * Σ_{neighbors} s_j
全体エネルギーを計算する場合は `total_energy` を使う。
"""
function local_energy(spins::AbstractMatrix, i::Integer, j::Integer; J=1)
    s = spins[i, j]
    return -J * s * sum_neighbors(spins, i, j)
end


"""
total_energy(spins; J=1)

格子全体のハミルトニアン H = -J Σ_{<ij>} s_i s_j を返す。
`local_energy` をすべてのサイトで合計すると各結合が二重に数えられるため、最後に 1/2 を掛けて調整する。
"""
function total_energy(spins::AbstractMatrix; J=1)
    L1, L2 = size(spins)
    @assert L1 == L2 "square lattice expected"
    L = L1
    total = zero(eltype(spins))
    for i in 1:L, j in 1:L
        total += local_energy(spins, i, j; J=J)
    end
    return total / 2
end


"""
magnetization(spins)

格子 `spins` の1サイト当たりの磁化を返す（Float）。
"""
function magnetization(spins::AbstractMatrix)
    N = length(spins)
    return sum(spins) / float(N)
end


"""
delta_energy_flip(spins, i, j; J=1)

サイト (i,j) のスピンを反転したときのエネルギー差 ΔE を局所的に計算する。
式: ΔE = E_new - E_old = 2 * J * s_i * Σ_neighbors s_j
全体再計算を避けて局所的に ΔE を得るためのヘルパー関数。
"""
function delta_energy_flip(spins::AbstractMatrix, i::Integer, j::Integer; J=1)
    s = spins[i, j]
    h = sum_neighbors(spins, i, j)
    return 2 * J * s * h
end


"""
fisher_yates!(rng, v)

インプレースの Fisher--Yates シャッフル。`shuffle!` の代替で割当を出さないことを狙う。
"""
function fisher_yates!(rng::AbstractRNG, v::Vector{Int})
    n = length(v)
    for k in n:-1:2
        j = rand(rng, 1:k)
        v[k], v[j] = v[j], v[k]
    end
    return v
end


# --- Convenience methods that accept an IsingState ---
function magnetization(state::IsingState)
    return magnetization(state.spins)
end

function total_energy(state::IsingState; J=nothing)
    J = J === nothing ? state.J : J
    return total_energy(state.spins; J=J)
end

function delta_energy_flip(state::IsingState, i::Integer, j::Integer; J=nothing)
    J = J === nothing ? state.J : J
    return delta_energy_flip(state.spins, i, j; J=J)
end

"""
flip!(state, i, j)

状態 `state` の (i,j) スピンをその場で反転する（副作用あり）。
"""
function flip!(state::IsingState, i::Integer, j::Integer)
    state.spins[i, j] = -state.spins[i, j]
    return nothing
end


"""
metropolis_trial!(state, i, j)

サイト `(i,j)` に対してメトロポリスの1試行を行う。
受容されたらスピンを反転し `true` を返す。棄却されたら `false` を返す。
"""
function metropolis_trial!(state::IsingState, i::Integer, j::Integer)
    ΔE = delta_energy_flip(state, i, j)
    if ΔE <= 0 || rand(state.rng) < exp(-state.beta * float(ΔE))
        flip!(state, i, j)
        return true
    else
        return false
    end
end


"""
metropolis_step!(state)

ランダムに1サイト選んでメトロポリスの1試行を行う。受容なら `true` を返す。
"""
function metropolis_step!(state::IsingState)
    L = size(state.spins, 1)
    i = rand(state.rng, 1:L)
    j = rand(state.rng, 1:L)
    return metropolis_trial!(state, i, j)
end


"""
metropolis_sweep!(state; random_order=false)

L x L のサイトに対して 1 スイープ分のメトロポリス試行を行う。
デフォルトは決定的（行優先）走査で割当を抑える。
`random_order=true` を指定するとランダム順に N=L^2 回試行する。
関数は受容された更新数を返す（整数）。
"""
function metropolis_sweep!(state::IsingState; random_order::Bool=false)
    L = size(state.spins, 1)
    N = L * L
    accepts = 0
    if random_order
        fisher_yates!(state.rng, state.idx)
        for k in state.idx
            i = state.i_idx[k]
            j = state.j_idx[k]
            accepts += metropolis_trial!(state, i, j) ? 1 : 0
        end
    else
        for i in 1:L, j in 1:L
            accepts += metropolis_trial!(state, i, j) ? 1 : 0
        end
    end
    return accepts
end



