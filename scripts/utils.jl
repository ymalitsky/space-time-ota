
function foldername(prefix, N_S, N_R, deg, P_max, seed, λ, n_iter)
    return prefix *
           "N_S=$(N_S)_N_R=$(N_R)_deg=$(deg)_P_max=$(P_max)/seed=$(seed)_lambda=$(λ)/n_iter=$(n_iter)"
end

function foldername_dict(prefix, d)
    s = prefix
    for key in keys(d)
        v = d[key]
        s *= "$(key)=$(v)_"
    end
    return s
end
