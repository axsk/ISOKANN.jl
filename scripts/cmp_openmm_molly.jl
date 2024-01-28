using ISOKANN

function sims(;
  temp=300.0,
  gamma=1.0,
  dt=0.002,
  T=1.0
)

  @show steps = ceil(Int, T / dt)

  pdb = "$(@__DIR__)/../data/alanine-dipeptide-nowater av.pdb"
  omm = OpenMMSimulation(; pdb, steps, temp, friction=gamma, features=1:66, step=dt)


  mol = MollyLangevin(; sys=PDB_ACEMD(),
    temp=temp, gamma=gamma, dt=dt, T=T)

  return (; omm, mol)
end

function test(n=1; kwargs...)
  oom, mol = sims(; kwargs...)
  x0 = reshape(getcoords(mol), :, 1)

  # it seems openmm is by factor 100 slower
  x1 = @time propagate(mol, x0, n)
  x2 = @time propagate(oom, x0, n)


  p1 = scatter_ramachandran(reshape(x1, :, n)) |> display
  p2 = scatter_ramachandran(reshape(x2, :, n)) |> display

  return NamedTuple(Base.@locals)
end