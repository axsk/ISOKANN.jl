### High-level workflow runners and dashboards

function run_metadynamics!(iso; generations=100, iter=100, plots=[], mdargs...)
    for _ in 1:generations
        @time adaptive_metadynamics(iso; mdargs...)
        @time run!(iso, iter)
        if plots != false
            p = metadynamics_dashboard(iso; mdargs...)
            display(p)
            push!(plots, p)
        end
    end
    return (; iso, plots)
end

function adaptive_metadynamics(iso; deposit=OpenMM.steps(iso.data.sim), x0=coords(iso)[:, end], mdargs...)
    md = MetadynamicsSimulation(iso; mdargs...)
    t = trajectory(md; x0, saveevery=deposit)
    @assert norm(t.values[:, end]) < 100
    xnew = values(t)
    addcoords!(iso, xnew)
    return (; t, md, xnew)
end


function metadynamics_dashboard(iso; mdargs...)
    l = @layout [[a; b] c{0.3w}]
    p1 = scatter_ramachandran(iso)
    scatter!(p1, [ISOKANN.phi(coords(iso)[:, end])], [ISOKANN.psi(coords(iso)[:, end])])
    p2 = plot_training(iso)
    p3 = plot_profile(MetadynamicsSimulation(iso; mdargs...), iso)
    return plot(p1, p3, p2, layout=l, size=(800, 800))
end


function run_kde_dash!(iso; generations=1, plots=[], kwargs...)
    for _ in 1:generations
        ISOKANN.run_kde!(iso; generations=1, kwargs...)
        p1 = scatter_ramachandran(iso)
        p2 = plot_training(iso)
        p = plot(p1, p2, layout=(1, 2), size=(800, 800))
        display(p)
        push!(plots, p)
    end
    return plots
end

function run_both!(iso; generations=100, samples_kde=1, iter=100, plots=[])
    for _ in 1:generations
        run_kde!(iso; generations=1, kde=samples_kde, iter)
        run_metadynamics!(iso; generations=1, iter, plots)
    end
end

function reactivepath_save(iso; path="reactive_path", kwargs...)
    hsh = hash(iso, hash(kwargs))
    ix = save_reactive_path(iso, sigma=0.03, minjump=0.02, maxjump=0.1, out="$path-$hsh.pdb"; kwargs...)
    OpenMM.potential(iso.data.sim, coords(iso)[:, ix]) |> vec |> plot
    savefig("$path-$hsh.png")
    scatter_ramachandran(coords(iso)[:, ix], x -> chicoords(iso, x))
    savefig("$path-$hsh.png")
end

function makeanim(ps, filename)
    a = Animation()
    for p in ps
        frame(a, p)
    end
    mp4(a, filename)
end
