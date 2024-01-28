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

function adapt_omm(;
    steps=100,
    sim=OpenMMSimulation(; steps),
    transform=TransformShiftscale(),
    n=1,
    nx=10,
    ny=5,
    nd=1,
    lr=1e-3,
    epochs=50,
    nkoop=20,
    nlearn=10,
    nres=10
)
    global iso
    @time iso = iso2(; sim, transform, n, nx, ny, nd, lr)
    @time iso = isosteps(iso, nkoop, nlearn)

    for e in 1:epochs
        @time (xs, ys) = adddata((iso.xs, iso.ys), iso.model, iso.sim, nres)
        iso = (; iso..., xs, ys,)
        @time iso = isosteps(iso, nkoop, nlearn)
        #plot(sort(vec(iso.target))) |> display
    end

    return Base.@locals
end