# the idea: take the derivative of the chi function as force field for an
# "average transition path"

using Zygote
using OrdinaryDiffEq

function force(sim, x, chi, direction=1, orth=0.01)
    # setcoords
    sys = setcoords(sim.sys, x)
    fs = Molly.forces(sys, find_neighbors(sys))
    fs = ustrip(reduce(vcat, fs))
    fc = Zygote.gradient(x->first(chi(x)), x)[1]

    fso = fs - fc * dot(fs, fc) / dot(fc, fc)
    dot(fso, fc)

    f = -fc #+ fso * 0.001
    f = direction * fc / norm(fc)^2 + fso * orth
    collect(f)

end

function transitionpath(sim, x0, chi, orth = 0.01; solver=Rosenbrock23(), kwargs...)
    direction = (first(chi(x0))) > 0.5 ? -1 : 1
    prob = ODEProblem((x,p,t)->force(sim, x, chi, direction, orth), x0, (0,1))
    sol = solve(prob, solver; kwargs...)
    #return reduce(hcat, sol.u)
end
