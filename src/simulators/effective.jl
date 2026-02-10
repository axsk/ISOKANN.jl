# Effective dynamics

function force(z, xi, xs::AbstractVector, sim)
    zs = xi.(xs)
    ws = kde(z, zs) # normalized!

    σ = constants(sim).sigma
    
    for x in xs
        F = OpenMM.force(sim, x) # force == -∇V
        Δxi = laplaces(xi, x)
        ∇xi = jac(xi, x)
        @show b = @. ws * (σ^2 / 2 * Δxi + F * ∇xi)
    end
end



# eq (15,16) in Legoll, Lelievre, (8) in Sikorski, Donati,  Weber, Schütte, (40) in Zhang, Hartmann, Schütte [2016]
function b_and_sigma(x, xi, sim)
    sigma = OpenMM.constants(sim).sigma
    F = OpenMM.force(sim, x) # force == -∇V
    H = laplaces(xi, x, sigma)
    J = jac(xi, x)
    b = H .+ J * F
    # ̃A = ̃σ ̃σ' = J A J' = J σ σ' * J' => 
    s = J .* sigma'
    sigtilde = s * s'

    return b, sigtilde, cholesky(sigtilde).L
end

import Zygote

jac(xi, x) = Zygote.jacobian(xi, x) |> only

raw"""
    laplaces(xi, x, sigma; n=length(xi(x)))
 
Geometric laplacions of components of xi:

laplaces(\xi, x)_k = \sum_i \frac{\sigma^2}{2} \frac{d^2 \xi_k(x)}{dx_j^2}

"""
function laplaces(xi, x, sigma; n=length(xi(x)))
    ntuple(n) do i
        H = only(Zygote.diaghessian(x->xi(x)[i], x))
        sum(@. sigma^2/2 * H)
    end
end
