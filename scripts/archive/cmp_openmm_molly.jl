using ISOKANN
using ISOKANN: getcoords

function sims(;
  temp=300.0,
  gamma=1.0,
  dt=0.002,
  T=1.0,
  pdb="$(@__DIR__)/../data/alanine-dipeptide-nowater.pdb"
)

  @show steps = ceil(Int, T / dt)


  ff = "ff99SBildn.xml"
  #ff = "amber14-all.xml"

  ffm = joinpath(ISOKANN.molly_data_dir, "force_fields", "ff99SBildn.xml")
  ffo = ["amber14-all.xml"]
  ffo = [ffm]

  omm = OpenMMSimulation(; pdb, forcefields=ffo, steps, temp, friction=gamma, features=1:66, step=dt)

  ff = ISOKANN.Molly.MolecularForceField(ffm)
  sys = ISOKANN.System(pdb, ff,
    rename_terminal_res=false, # this is important,
    #boundary = CubicBoundary(Inf*u"nm", Inf*u"nm", Inf*u"nm")  breaking neighbor search
  )

  mol = MollyLangevin(; sys,
    temp=temp, gamma=gamma, dt=dt, T=T)

  return (; omm, mol)
end

function test(n=1; kwargs...)
  oom, mol = sims(; kwargs...)
  x0 = reshape(getcoords(mol), :, 1)

  # it seems openmm is by factor 100 slower
  x1 = @time propagate(mol, x0, n)
  x2 = @time propagate(oom, x0, n)


  p1 = scatter_ramachandran(reshape(x1, :, n))
  p2 = scatter_ramachandran(reshape(x2, :, n))

  return NamedTuple(Base.@locals)
end