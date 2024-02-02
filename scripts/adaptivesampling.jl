using ISOKANN
using ISOKANN: diala2, adddata, isosteps

# proof of concept for adaptive sampling with isokann 2
function adapt_diala()
    ia = IsoRun()
    iso = diala2()
    ndata = adddata(iso.data, iso.model, ia.sim, 10; renormalize=true)
    isosteps()
end

using ISOKANN: OpenMMSimulation, TransformShiftscale, iso2
using Plots

function adapt_omm(;
    steps=100,
    sim=OpenMMSimulation(pdb="data/alanine-dipeptide-nowater av.pdb", steps=steps, features=1:22),
    transform=TransformShiftscale(),
    n=1,
    nx=10,
    ny=5,
    nd=1,
    lr=1e-3,
    epochs=1,
    nkoop=20,
    nupdate=10,
    nres=10
)
    global iso
    @time iso = iso2(; sim, transform, n, nx, ny, nd, lr)
    @time iso = isosteps(; iso, nkoop, nupdate)

    for e in 1:epochs
        iso = adddata(iso, nres)
        #@time (xs, ys) = adddata((iso.xs, iso.ys), iso.model, iso.sim, nres)
        #iso = (; iso..., xs, ys,)
        @time iso = isosteps(iso)
        plot(sort(vec(iso.target))) |> display
    end

    return (; NamedTuple(Base.@locals)..., iso...)
end

function ISOKANN.adddata(iso::NamedTuple, nres=nothing)
    nres === nothing && (nres = iso.nkoop)
    @time (xs, ys) = adddata((iso.xs, iso.ys), iso.model, iso.sim, nres)
    iso = (; iso..., xs, ys, nres)
end
