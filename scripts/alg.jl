using LinearAlgebra



function stop(it, err, maxit, tol)
    if it > maxit || err <= tol
        return false
    end
end

function linesearch_initial(oracle_f, g, prox_g, x0, α; verbose = false)
    obj_x, grad_fx0 = oracle_f(x0)
    largestep = true
    for i in range(1, 100)
        x1 = prox_g(x0 - α * grad_fx0, α)
        obj_x1, grad_fx1 = oracle_f(x1)
        if i == 1 && isapprox(x0, x1)
            verbose && println("Congrats: initial x0 is a solution")
        else
            L = norm(grad_fx1 - grad_fx0) / norm(x1 - x0)
            if α * L > 2
                # decreasing step
                largestep = false
                verbose && println("/2")
                α *= 0.5
            else
                if α * L <= 2 && largestep
                    # increasing step, aggresively
                    α *= 10
                    verbose && println("*10")
                    if α > 10
                        return α, x1, grad_fx0, grad_fx1, obj_x, obj_x1, i
                        break
                    end

                else
                    return α, x1, grad_fx0, grad_fx1, obj_x, obj_x1, i
                    break
                end
            end
        end
    end

end

function collect_history(history_dic, data_dic)
    for key in keys(history_dic)
        push!(history_dic[key], data_dic[key])
    end
    return history_dic
end

"""
    AdProxGrad()

Adaptive proximal gradient method that computes stepsize on the fly. Based on Y. Malitsky, K. Mishchenko 'Adaptive Gradient Descent without Descent', https://arxiv.org/abs/1910.09529

oracle_f: returns function value and its gradient
prox_g: returnes a proximal operator, in our case it will be a projection
x0: initial point
maxit: the maximal number of iterations
tol: required accuracy
lns = true: means that the we run linesearch on the first iteration. This is needed to understand in which range the first stepsize should be. 
"""

function AdProxGrad(
    oracle_f,
    g,
    prox_g,
    x0;
    maxit = 1000,
    tol = 1e-9,
    stop = "res",
    lns = true,
    verbose = false,
    ver = 1,
    track = ["res", "obj", "grad", "steps"],
    fixed_step = 1e-2,
)
    x_prev = x0
    θ = 1.0 / 3
    if lns
        α_prev, x, grad_prev, grad_x, obj_prev, obj_x, lns_iter =
            linesearch_initial(oracle_f, g, prox_g, x0, 1e-2)
        verbose &&
            println("Linesearch found initial stepsize $α_prev in $lns_iter iterations")
    else
        α_prev = 1e-6
        verbose && println("No linesearch, initial stepsize is ", α_prev)
        obj_prev, grad_prev = oracle_f(x_prev)
        x = prox_g(x_prev - α_prev * grad_prev, α_prev)
        obj_x, grad_x = oracle_f(x)
    end
    dict = Dict(
        "res" => [norm(x - x_prev) / α_prev],
        "obj" => [obj_x],
        "grad" => norm.([grad_prev, grad_x]),
        "steps" => [α_prev],
    )
    history_dic = filter(p -> p.first in track, dict)

    i = 1
    for i in range(1, maxit)
        L = norm(grad_x - grad_prev) / norm(x - x_prev)
        if ver == 1
            α = min(sqrt(1 + θ) * α_prev, 1 / (sqrt(2) * L))
        elseif ver == 2
            α = min(sqrt(2 / 3 + θ) * α_prev, α_prev / sqrt(max(2 * α_prev^2 * L^2 - 1, 0)))
        elseif ver == 0
            α = fixed_step
        end

        θ = α / α_prev
        x_prev, grad_prev, α_prev = x, grad_x, α
        x = prox_g(x - α * grad_x, α)

        residual = norm(x_prev - x) / α
        obj_x, grad_x = oracle_f(x)
        current_info = Dict(zip(track, [residual, obj_x, norm(grad_x), α]))
        collect_history(history_dic, current_info)
        if current_info[stop] <= tol
            verbose && println("The algorithm reached required accuracy in $i iterations")
            break
        end

    end

    return x, history_dic
end
