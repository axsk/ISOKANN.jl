#!/home/htc/bzfsikor/bin/julia
using Pkg
Pkg.precompile()

using ISOKANN
using Optimisers
using ISOKANN: pairnetn
using JLD2

nocallback(;kwargs...) = nothing

id = get(ENV, "SLURM_PROCID", 0)
job = get(ENV, "SLURM_JOB_ID", 0)

id = "$job-$id"


#
# what we have so far:
# isoreference:
# nk = 128, nd=10_000, nres=50, np=2, nl=5, nx=100, ny=20, opt=1e-4
# isoreference-6356545:
# nk = 128, nd=10_000, nres=50, np=2, nl=5, nx=100, ny=20, opt=1e-3
# isoreference-6366008:
# nk = 256, nd=20_000, nres=100, np=2, nl=5, nx=100, ny,20, opt=1e-3


function isoreference(id=id, save=true)
    println("computing reference isokann solution $id")
    iso = ISORun(
        sim=MollyLangevin(sys=PDB_ACEMD(), dt=1e-4),
        model=pairnetn(22,4),
        opt = Optimisers.Adam(1e-3),
        nd=20_000,
        ny=20,
        nk=256,
        nres=100,
    )
    @time run!(iso, callback=nocallback)
    save && JLD2.save("isoreference-$id.jld2", "iso", iso)
    return iso
end

function isodefault(id=id)
    println("computing default isokann solution")
    iso = ISORun()
    @time run!(iso, callback=nocallback)
    return iso
end
