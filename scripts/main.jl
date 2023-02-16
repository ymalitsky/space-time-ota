using JLD2
using CSV
using OrderedCollections
using DataFrames
include("problem.jl")
include("plotting.jl")
include("alg.jl")


"""
    main()

Main function to generate all data and plots. First we solve optimization problem for each T, SNR, and seed. We save all data to the corresponding files. If `avg_over_h` is `true`, we average our reconstructed error by averaging over h.  It also generates correspondent plots.
"""
function main(T_list, SNR_list, n_iter, seeds, N_S, N_R, deg, P_max, λ; avg_over_h = false)
    M = 10000  # averages over noise M times
    @time for seed in seeds
        df = run_exp(T_list, SNR_list, N_S, N_R, P_max, λ, deg, seed, n_iter, M)
        println("-----Seed $seed is finished-----")
    end
    # Find the error after averaging over h 
    if avg_over_h
        df = averaging_over_h(N_S, N_R, P_max, λ, deg, seeds, n_iter, M; write = true)
    end
    return df
end

"""
    run_exp()

Function that runs single experiment for a fixed seed and for each T and SNR in the list. Saves the data and generates plots. 
"""
function run_exp(T_list, SNR_list, N_S, N_R, P_max, λ, deg, seed, n_iter, M)
    folder = foldername_dict(
        "./data/",
        OrderedDict(
            "Ns" => N_S,
            "Nr" => N_R,
            "deg" => deg,
            "Pmax" => P_max,
            "lambda" => λ,
            "iter" => n_iter,
        ),
    )
    p = Problem(deg, N_S, N_R, P_max; λ = λ, seed = seed)
    save(folder * "/seed=$(seed)/problem.jld2", "problem", p)
    nodenames = [:T, :SNR, :σ, :error, :error_simple_ota, :res]
    df = DataFrame([name => Float64[] for name in nodenames])
    res = 0.0
    for T in T_list
        prox_g, g, oracle_f, X0 = algorithmic_ingredients(p, T)
        filename = folder * "/seed=$(seed)/X_T=$T.jld2"
        if isfile(filename)
            println("File is already here")
            load_data = load(filename)
            X1 = load_data["X"]
            res = load_data["history"]["res"][end]
        else
            X1, history = AdProxGrad(
                oracle_f,
                g,
                prox_g,
                X0;
                maxit = n_iter,
                tol = 1e-5,
                stop = "res",
                lns = true,
                verbose = true,
                track = ["res", "obj", "grad", "steps"],
            )
            res = history["res"][end]
            println("Problem T=$T, residual=$res")
            save(filename, Dict("X" => X1, "history" => history))
        end
        for SNR in SNR_list
            σ = round(sqrt(P_max / SNR), digits = 3)
            error = signal_error_avg(p, X1; σ = σ, M = M, use_seed = true)
            println(T, ": ", error)
            error_simple_ota = signal_error_standard_ota(p, SNR)
            push!(df, [T, SNR, σ, error, error_simple_ota, res])
        end
    end
    output = folder * "/seed=$(seed)/errors_avg_over_noise.csv"
    CSV.write(output, df)
    folderplot =
        foldername_dict(
            "./plots/",
            OrderedDict(
                "Ns" => N_S,
                "Nr" => N_R,
                "deg" => deg,
                "Pmax" => P_max,
                "lambda" => λ,
                "iter" => n_iter,
            ),
        ) * "/seed=$seed"
    plots_from_df(df, folderplot)
    return df
end

"""
    averaging_over_h()

Read data from files and compute the averaged errors. Also generate correspondent plots.
"""
function averaging_over_h(N_S, N_R, P_max, λ, deg, seeds, n_iter, M; write = true)
    nodenames = [:T, :SNR, :σ, :error, :error_simple_ota, :res]
    df = DataFrame([name => Float64[] for name in nodenames])
    folder = foldername_dict(
        "./data/",
        OrderedDict(
            "Ns" => N_S,
            "Nr" => N_R,
            "deg" => deg,
            "Pmax" => P_max,
            "lambda" => λ,
            "iter" => n_iter,
        ),
    )
    n = length(seeds)
    error = 0.0
    error_simple_ota = 0.0
    res = 0.0
    for (i, seed) in enumerate(seeds)
        filename = folder * "/seed=$(seed)/errors_avg_over_noise.csv"
        df_file = CSV.read(filename, DataFrame)
        if i == 1
            df = copy(df_file)
            nn = size(df_file.error)
            error = zeros(nn)
            error_simple_ota = zeros(nn)
            res = zeros(nn)
        end
        error .+= df_file.error
        error_simple_ota .+= df_file.error_simple_ota
        res .+= df_file.res
        println(i, ": ", norm(error), ",   ", norm(error_simple_ota))
    end
    df.error = error / n
    df.error_simple_ota = error_simple_ota / n
    df.res = res / n
    d = OrderedDict(
        "Ns" => N_S,
        "Nr" => N_R,
        "deg" => deg,
        "Pmax" => P_max,
        "lambda" => λ,
        "iter" => n_iter,
    )
    write && CSV.write(folder * "/errors_avg_seeds=$(seeds[1])-$(seeds[end]).csv", df)
    folderplot = foldername_dict("./plots/", d) * "/avg_seeds=$(seeds[1])-$(seeds[end])"
    plots_from_df(df, folderplot)
    return df
end

"""
    averaging_over_h2()

Alternative implementation that just averages all csv files.
"""

function averaging_over_h2(T_list, N_S, N_R, P_max, λ, deg, seeds, n_iter; write = true)
    folder = foldername_dict(
        "./data/",
        OrderedDict(
            "Ns" => N_S,
            "Nr" => N_R,
            "deg" => deg,
            "Pmax" => P_max,
            "lambda" => λ,
            "iter" => n_iter,
        ),
    )
    n = length(seeds)
    filenames = [folder * "/seed=$(seed)/errors_avg_over_noise.csv" for seed in seeds]
    df_list = [CSV.read(filename, DataFrame) for filename in filenames]
    # In case I did some extra experiments for T that I don't intend to include
    df_list = filter.(row -> row.T in T_list, df_list)
    df = reduce(.+, df_list) ./ n
    d = OrderedDict(
        "Ns" => N_S,
        "Nr" => N_R,
        "deg" => deg,
        "Pmax" => P_max,
        "lambda" => λ,
        "iter" => n_iter,
    )
    write && CSV.write(folder * "/check_errors_avg_seeds=$(seeds[1])-$(seeds[end]).csv", df)
    folderplot =
        foldername_dict("./plots/", d) * "/check_avg_seeds=$(seeds[1])-$(seeds[end])"
    plots_from_df(df, folderplot)
    return df
end

"""
    compare_energy_consumptions()

Compare how much energy two approaches use
"""

function compare_energy_consumptions(N_S, N_R, deg, P_max, λ, n_iter, SNR; seed = 1)
    folder = foldername_dict(
        "./data/",
        OrderedDict(
            "Ns" => N_S,
            "Nr" => N_R,
            "deg" => deg,
            "Pmax" => P_max,
            "lambda" => λ,
            "iter" => n_iter,
        ),
    )
    filename_problem = folder * "/seed=$(seed)/problem.jld2"
    problem = load(filename_problem)["problem"]
    BM = problem.BM
    h = problem.h
    s = problem.s
    C = problem.C
    σ = sqrt(P_max / SNR)

    for (k, T) in enumerate(T_list)
        filename_sol = folder * "/seed=$(seed)/X_T=$T.jld2"
        X1 = load(filename_sol)["X"]
        energy_ota = 0.0
        if k == 1
            for j = 1:N_R
                w = 1 / sum(BM[:, j])
                mask = findall(>(0), BM[:, j])
                η = sqrt(P_max / deg) / w * minimum(abs.(h[mask, j]) ./ abs.(s[mask]))
                b = η * w ./ h[mask, j]
                energy_ota += dot(norm.(b) .^ 2, norm.(s[mask]) .^ 2)
            end
            println("Standard OtA energy consumption = ", energy_ota)
            println("Constraint N_s * P_max = ", N_S * P_max)
        end
        P1, Q1 = X1
        P2 = projball_by_cols(P1, C)
        println("Constraints violation, ", norm(P1 - P2))
        error = dot(norm.(eachcol(P1)) .^ 2, abs.(s) .^ 2)
        energy_new = 0.0
        for i = 1:N_S
            temp = norm(P1[:, i])^2 * abs(s[i])^2
            energy_new += temp
        end
        println("New approach, T=$T = ", error)
    end
end


function compute_optimized_ota_matrix(deg, N_S, N_R, P_max; seed = 1, λ = 0.1, SNR = 10)
    p = Problem(deg, N_S, N_R, P_max; seed = seed, λ = λ)
    h, BM, θ, s = p.h, p.BM, p.θ, p.s
    σ = sqrt(P_max / SNR)
    A = zeros(N_S, N_R)
    for i = 1:N_S
        for j = 1:N_R
            if BM[i, j] == 1.0
                w = 1 / sum(BM[:, j])
                A[i, j] = σ^2 / N_R * w^2 * abs(s[i])^2 / abs(h[i, j])^2
            end
        end
    end
    sqrtA = sqrt.(A)
    P_opt_ota = P_max * sqrtA ./ sum(sqrtA, dims = 2)
    lower_bound = maximum(sum(sqrt.(A), dims = 2))^2 / P_max
    # if we substitute p_{ij} = k_i sqrt(a_ij)
    D = sqrtA .* sum(sqrtA, dims = 2) / P_max
    upper_bound = sum(maximum(D, dims = 1))
    return lower_bound, upper_bound
end


# define parameters for experiment. Below is what was used for the paper


T_list = [14:2:30; 35; 40; 45]
SNR_list = [1.0, 10.0, 100.0]
N_S = 50
N_R = 30
deg = 20
P_max = 10.0
seeds = collect(1:100)
λ = 0.1
n_iter = 100000



# Uncomment the lines below to generate all the plots. It takes some time, since there are too many optimization problems to solve. A good thing is that all data will be saved on the disk, so next time it you need to tune something, it will be much faster.


# Generate all data and solve optimization problems for space-time approach.

#main(T_list, SNR_list, n_iter, seeds, N_S, N_R, deg, P_max, λ; avg_over_h = true)


# Make histograms
# make_histograms(30, deg, N_S, N_R, P_max, 1, λ)

# Plots with extra line that shows (almost) optimal OtA for SNR=1,10, 100
# optimal_MSE_average_over_h(deg, N_S, N_R, P_max, seeds; λ = 0.1, SNR=1, simple_ota=false)

# optimal_MSE_average_over_h(deg, N_S, N_R, P_max, seeds; λ = 0.1, SNR=10, simple_ota=false)

# optimal_MSE_average_over_h(deg, N_S, N_R, P_max, seeds; λ = 0.1, SNR=100, simple_ota=false)
