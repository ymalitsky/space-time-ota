using Plots
using StatsPlots
using Plots.Measures
using LaTeXStrings
using DataFrames
include("utils.jl")


plot_font = "Computer Modern"
default(
    fontfamily = plot_font,
    linewidth = 6,
    framestyle = :box,
    markersize = 11,
    grid = true,
    xtickfontsize = 18,
    ytickfontsize = 18,
    guidefontsize = 18,
    legendfontsize = 16,
    titlefontsize = 19,
)

"""
    plots_from_df(df, descr, n_iter, M; simple_ota=true)

If `simple_ota = true` then plot two lines: space-time approach and the standard OtA, otherwise plot only the former.
"""

function plots_from_df(df, folderplot; simple_ota = true)
    for snr in unique(df.SNR)
        df2 = df[df.SNR.==snr, :]
        if snr == 1.0
            label_y = "MSE"
        else
            label_y = ""
        end
        @df df2 scatter(
            :T,
            :error,
            xlabel = L"$T$",
            ylabel = label_y,
            labels = "proposed approach",
        )
        error_simple_ota = unique(df2.error_simple_ota)
        println("Simple OtA: ", error_simple_ota)
        simple_ota && hline!([error_simple_ota], label = "standard OtA")
        title!(L"P_{\max}/\sigma^2" * "=$(snr)")
        if !ispath(folderplot)
            println("No such folder, I will make one")
            mkpath(folderplot)
        end
        savefig(folderplot * "/snr=$(snr).pdf")
    end
end


function make_histograms(T, deg, N_S, N_R, P_max, seed, λ; SNR = 10)
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
    problem = load(folder * "/seed=$(seed)/problem.jld2")["problem"]
    filename_sol = folder * "/seed=$(seed)/X_T=$T.jld2"
    X1 = load(filename_sol)["X"]
    s = problem.s
    h = problem.h
    BM = problem.BM
    P1, Q1 = X1
    P_opt_ota, _, _ =
        compute_optimized_ota_matrix(deg, N_S, N_R, P_max; seed = seed, λ = λ, SNR = SNR)
    consumed_power_new = zeros(N_S)
    consumed_power_ota = zeros(N_S)
    consumed_power_ota_optimized = zeros(N_S)
    for i = 1:N_S
        consumed_power_new[i] = norm(P1[:, i])^2 * abs(s[i])^2
    end

    for j = 1:N_R
        w = 1 / sum(BM[:, j])
        mask = findall(>(0), BM[:, j])
        η = sqrt(P_max / deg) / w * minimum(abs.(h[mask, j]) ./ abs.(s[mask]))
        b = η * w ./ h[mask, j]
        consumed_power_ota[mask] += (abs.(b .* s[mask])) .^ 2
        η_opt_ota =
            (1 ./ w) *
            minimum(sqrt.(P_opt_ota[mask, j]) .* abs.(h[mask, j]) ./ abs.(s[mask]))
        b_opt_ota = η_opt_ota * w ./ h[mask, j]
        consumed_power_ota_optimized[mask] += (abs.(b_opt_ota .* s[mask])) .^ 2
    end


    folderhist =
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
        ) *
        "/seed=$seed" *
        "_T=$T"

    if !ispath(folderhist)
        println("No such folder, I will make one")
        mkpath(folderhist)
    end
    histogram(
        min.(consumed_power_new, 9.999),
        bins = 10,
        legend = false,
        xlabel = "consumed power",
        color = 1,
        ylabel = "number of senders",
        bottom_margin = 0.5cm,
    )
    savefig(folderhist * "hist_new.pdf")
    histogram(
        min.(consumed_power_ota, 9.999),
        bins = 10,
        color = 2,
        legend = false,
        xlabel = "consumed power",
        ylabel = "number of senders",
        bottom_margin = 0.5cm,
    )
    savefig(folderhist * "hist_ota.pdf")
    histogram(
        (consumed_power_ota_optimized),
        bins = 10,
        legend = false,
        xlabel = "consumed power",
        color = 3,
        ylabel = "number of senders",
        bottom_margin = 0.5cm,
    )
    savefig(folderhist * "hist_opt_ota.pdf")
    return consumed_power_ota, consumed_power_new, consumed_power_ota_optimized
end

function optimal_MSE_average_over_h(
    deg,
    N_S,
    N_R,
    P_max,
    seeds;
    λ = 0.1,
    SNR = 10,
    simple_ota = false,
)
    n = length(seeds)
    lower_bound = zeros(n)
    upper_bound = zeros(n)
    for (i, seed) in enumerate(seeds)
        lower_bound[i], upper_bound[i] = compute_optimized_ota_matrix(
            deg,
            N_S,
            N_R,
            P_max;
            seed = seed,
            λ = λ,
            SNR = SNR,
        )
    end
    mlb, mub = mean(lower_bound), mean(upper_bound)
    filename =
        foldername_dict(
            "./data/",
            OrderedDict(
                "Ns" => N_S,
                "Nr" => N_R,
                "deg" => deg,
                "Pmax" => P_max,
                "lambda" => λ,
                "iter" => n_iter,
            ),
        ) * "/errors_avg_seeds=$(seeds[1])-$(seeds[end]).csv"
    df = CSV.read(filename, DataFrame)
    df2 = df[df.SNR.==SNR, :]
    error = df2.error
    error_simple_ota = unique(df2.error_simple_ota)

    @df df2 scatter(
        :T,
        :error,
        xlabel = L"$T$",
        ylabel = "MSE",
        labels = "proposed approach",
    )
    simple_ota && hline!([error_simple_ota], label = "standard OtA", color = 2)
    hline!([mub], label = "optimized OtA", color = 3)
    folderplot = "./plots/optimized_ota"
    if !ispath(folderplot)
        println("No such folder, I will make one")
        mkpath(folderplot)
    end
    savefig(folderplot * "/snr=$(SNR)_standard_ota=$(simple_ota).pdf")
    return
end
