using LinearAlgebra
using Random
using StatsBase




include("utils.jl")

"""
    Problem

Structure that collects all information about the problem
...
#Arguments 
- `deg::Integer`: the degree of every node
- `N_S::Integer`: the number of senders
- `N_R::Integer`: the number of receivers
-  `BM`: biadjecency matrix, which is generated automatically from other data
- `W`: matrix defined by channel matrix `h` and biadjecency matrix `BM`
- `C`: vector of power constraints
- `h`: complex matrix that represents a channel gain for each edge and 0 otherwise. Generated from the complex normal distribution.
- `θ`: received signal, a vector of length N_R
-  `s`: a complex vector of length N_S, message to be sent. Generated randomly with an always fixed seed 2022. 
-  `P_max:`: power constraint, positive number
-  `seed`: seed to generate random `h`
-   `λ`: regularization parameter, a positive number

To define Problem structure we only need to pass `deg`, `N_S`, `N_R`, `P_max`
...
"""
struct Problem
    deg::Int64
    N_S::Int64
    N_R::Int64
    BM::Matrix{Float64}
    W::Matrix{ComplexF64}
    C::Vector{Float64}
    h::Matrix{ComplexF64}
    θ::Vector{ComplexF64}
    s::Vector{ComplexF64}
    P_max::Float64
    seed::Int64
    λ::Float64

    #constructor
    function Problem(deg, N_S, N_R, P_max; seed = 2022, λ = 0.1)
        type = ComplexF64
        #adjecency matrix is always the same 
        BM = biadjacency_matrix(N_S, N_R; deg = deg, seed = 2022, flag = "sender degree")
        Random.seed!(2022) # message s is always generated with the same seed
        s = randn(type, N_S)
        Random.seed!(seed)
        h = randn(type, N_S, N_R)
        h[BM.==0.0] .= Inf
        nbhs_R = sum(BM, dims = 1)
        W = 1 ./ (h .* nbhs_R)
        θ = [(BM' * s) ./ nbhs_R'...]
        C = P_max ./ abs.(s) .^ 2
        h[BM.==0.0] .= 0.0
        return new(deg, N_S, N_R, BM, W, C, h, θ, s, P_max, seed, λ)
    end
end


"""
    biadjacency_matrix(N_S, N_R; deg = 3, seed = 2022, flag = "sender degree")

Create a random biadjacency matrix for bipartite graph with `N_S` and `N_R` vertices. 

`deg` defines the fixed degree of the node. If `flag`= "sender degree", then each *sender's* node has degree `deg`. If `flag` = "receiver degree", then each *receiver's* node has a fixed degree `deg`. If `flag` is set to something else, then it returns just a random (0-1) matrix. 
"""
function biadjacency_matrix(N_S, N_R; deg = 3, seed = 2022, flag = "sender degree")
    BM = zeros(N_S, N_R)
    Random.seed!(seed)
    if flag == "sender degree"
        for i = 1:N_S
            ind = sample(1:N_R, deg, replace = false)
            BM[i, ind] .= 1
        end
    elseif flag == "receiver degree"
        for j = 1:N_R
            ind = sample(1:N_S, deg, replace = false)
            BM[ind, j] .= 1
        end
    else
        # if flag is different, then return some random 0-1 matrix
        BM = (sign.(randn(N_S, N_R) .+ 0.5) .+ 1) / 2
    end
    return BM
end


"""
    projball(x, r)

Project a vector `x` onto the ball with the center at origin and radius `r`
"""
function projball(x, r)
    norm_x = norm(x)
    if norm_x > r
        x *= r / norm_x
    end
    return x
end

"""
    projball_by_cols(P, c)

For every `j` project each column of a matrix `P` onto balls of radius \$\\sqrt c_j\$
"""
function projball_by_cols(P, c)
    n_cols = size(P)[2]
    for j = 1:n_cols
        P[:, j] = projball(P[:, j], sqrt(c[j]))
    end
    return P
end



"""
    algorithmic_ingredients(p::Problem, T)

Define all  ingredients for optimization algorithm AdProxGrad. This includes `prox_g, g, oracle_f, X_init`
"""
function algorithmic_ingredients(p::Problem, T)
    C = p.C
    function prox_g(X, α)
        X[1] = projball_by_cols(X[1], C)
        return X
    end
    g(x) = 0

    function oracle_f(X)
        λ, W = p.λ, p.W
        P, Q = X[1], X[2]
        res = transpose(P) * Q - W
        obj = norm(res)^2 + λ * norm(Q)^2
        grad_P = 2 * (conj(Q) * transpose(res))
        grad_Q = 2 * (conj(P) * res + λ * Q)
        return obj, [grad_P, grad_Q]
    end

    W, N_S, N_R, seed = p.W, p.N_S, p.N_R, p.seed
    Random.seed!(seed)
    P0 = rand(eltype(p.W), T, N_S)
    Q0 = rand(eltype(p.W), T, N_R)
    X_init = [P0, Q0]

    return prox_g, g, oracle_f, X_init
end



"""
    signal_error_avg(p::Problem, X_sol; σ = 1.0, M = 1000, use_seed = true)

For given problem `p` and the solution of optimization problem `X_sol = [P, Q]`, compute the average signal error \$\\frac{1}{N_R M} \\|\\hat \\theta - \\theta \\|^2\$. The *average* here because we average over `M` random instances of noise. That's why the final result is divided by `M` in the end.

"""
function signal_error_avg(p::Problem, X_sol; σ = 1.0, M = 1000, use_seed = true)
    # TODO: ugly! rewrite
    P, Q = X_sol
    h, BM, θ, s = p.h, p.BM, p.θ, p.s
    N_S, N_R = p.N_S, p.N_R
    T = size(X_sol[1])[1]
    if use_seed
        seed = p.seed
        Random.seed!(seed)
    end
    noise = randn(eltype(Q), (T, N_R, M)) .* σ
    err = 0
    θ_hat = zeros(eltype(θ), (length(θ), M))
    for m = 1:M
        Y = P * (h .* BM .* s) + noise[:, :, m]
        θ_hat[:, m] = [transpose(sum(Q .* Y, dims = 1))...]
    end
    err = 1.0 / (M * N_R) * norm(θ_hat .- θ, 2)^2
    return err
end






"""
    signal_error_standard_ota(p::Problem, SNR)

Compute the MSE for standard OtA approach.
"""
function signal_error_standard_ota(p::Problem, SNR)
    BM = p.BM
    P_max = p.P_max
    h = p.h
    s = p.s
    deg = p.deg
    σ = sqrt(P_max / SNR)
    N_S, N_R = size(BM)
    η = zeros(N_R)
    for j = 1:N_R
        w = 1 / sum(BM[:, j])
        mask = findall(>(0), BM[:, j])
        η[j] = sqrt(P_max / deg) / w * minimum(abs.(h[mask, j]) ./ abs.(s[mask]))
    end
    error = 1.0 / N_R * (sum(σ^2 ./ (η .^ 2)))
    return error
end
