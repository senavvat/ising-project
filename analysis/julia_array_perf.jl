# Julia 配列性能ミニチュートリアル
# Part 1: allocation, GC, in-place (.=)
# Part 2: column-major とアクセス順による性能差 (簡易 mul! 実験)

# --- Part 1: 短い解説 ---
# allocation が増えると何が起きるか:
# - ヒープ上の新しいメモリが確保され、メモリ割当が増える。
# - BenchmarkTools の出力に "allocs" / "allocated" 数値として現れる。
# - 多くの割当は CPU 時間を増やし、メモリ使用量を押し上げる。
#
# GC と allocation の関係:
# - 割当が増えると GC の発生頻度が上がる可能性がある（ヒープが満ちるため）。
# - GC が走るとプログラムは一時停止（停止時間は GC の種類と割当量に依存）。
# - したがって大量割当は遅延とスループット低下につながる。
#
# .=(インプレース更新) が効く理由:
# - 結果を既存配列に上書きすることで新しいヒープ割当を回避できる。
# - ブロードキャストのドット構文 (``out .= a .+ b``) はループを融合し、余分な一時配列を作らない。

# --- Part 1: 実験コード (割当の違いを見る) ---
try
    using BenchmarkTools
catch
    println("BenchmarkTools が見つかりません。実行するには: julia --project=. -e 'using Pkg; Pkg.add(\"BenchmarkTools\")'")
    exit()
end

# a + b を返す（割当あり）
function alloc_add(a::AbstractVector, b::AbstractVector)
    return a + b
end

# 結果を out に書き込む（インプレース、割当なしが期待）
function inplace_add!(out::AbstractVector, a::AbstractVector, b::AbstractVector)
    out .= a .+ b
    return out
end

# ベンチの実行関数
function bench_addition(n=10^6)
    a = rand(n)
    b = rand(n)
    out = similar(a)
    println("-- vector add, n=$(n) --")
    @btime alloc_add($a, $b)
    @btime inplace_add!($out, $a, $b)
    return nothing
end

# --- Part 2: column-major とアクセス順 ---
# Julia の配列は column-major（Fortran 風）。
# したがって列方向（row index を動かすアクセス）が連続アクセスになり高速。

# 良いループ順: j outer, k middle, i inner
function mul_good!(C::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix)
    n = size(A,1)
    @inbounds for j in 1:n
        for k in 1:n
            bk = B[k,j]
            for i in 1:n
                C[i,j] += A[i,k] * bk
            end
        end
    end
    return C
end

# 悪いループ順: i outer, k middle, j inner  (j を内側にすると C[i,j] の j 変化でストライドアクセスになる)
function mul_bad!(C::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix)
    n = size(A,1)
    @inbounds for i in 1:n
        for k in 1:n
            aik = A[i,k]
            for j in 1:n
                C[i,j] += aik * B[k,j]
            end
        end
    end
    return C
end

# ベンチ用ランナー
function bench_matmul(n=200)
    A = rand(n,n)
    B = rand(n,n)
    C1 = zeros(n,n)
    C2 = zeros(n,n)
    println("-- matmul comparison n=$(n) --")
    # warmup
    mul_good!(C1, A, B)
    mul_bad!(C2, A, B)
    C1 .= 0.0
    C2 .= 0.0
    @btime mul_good!($C1, $A, $B)
    @btime mul_bad!($C2, $A, $B)
    return nothing
end

# スクリプトとして実行されたときの簡単なデモ
if abspath(PROGRAM_FILE) == @__FILE__
    println("Part 1: allocation vs in-place")
    bench_addition(10^6)
    println()
    println("Part 2: column-major access order")
    bench_matmul(250) # 250x250 で差が見えやすい
end
