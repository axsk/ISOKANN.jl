using ISOKANN
using ISOKANN.IsoMu
using Random
using StatsBase
using ParameterSchedulers
using Optimisers

using Plots, JLD2

# TODO: update to current data dir
dataprefix = "/data/numerik/ag_cmd/trajectory_transfers_for_isokann/data/Feb19_data_for_paper_2024/des_prod_7UL4_159_251_disulfide_100ns_extension_replica__"

function papersuru(;
  n=3000,
  dataprefix=dataprefix,
  learnrate=1e-3,
  lrdecay=1e-2,
  regularization=1e-2,
  seed=1,
  minibatch=200,
  sigma=0.3,
  path=IsoMu.QuantilePath(0.01))

  Random.seed!(seed)
  ISOKANN.CUDA.seed!(seed) # note that GPU is not entierly deterministic

  links = [DataLink("$dataprefix$i") for i in 1:10]

  mu = isokann(links; learnrate, regularization, minibatch)

  if true
    nbatches = size(mu.iso.data[1], 2) / minibatch
    lrdecay = lrdecay^(1 / (n * nbatches))
    schedule = Scheduler(AdamRegularized, (Exp(learnrate, lrdecay), x -> regularization))
    #schedule = AdamW(eta=learnrate, lambda=regularization)
    mu.iso.opt = schedule
  end

  IsoMu.train!(mu, n)
  #return mu.iso
  #chis = mu.iso.model(mu.iso.data[1])
  #@show StatsBase.mean_and_std(chis), extrema(chis)
  #=
    #Flux.adjust!(mu.iso.opt, eta=learnrate * 0.1)
    train!(mu, n)
    chis = mu.iso.model(mu.iso.data[1])
    @show StatsBase.mean_and_std(chis), extrema(chis)

    #Flux.adjust!(mu.iso.opt, eta=learnrate * 0.01)
  train!(mu, n)
    chis = mu.iso.model(mu.iso.data[1])
    @show StatsBase.mean_and_std(chis), extrema(chis)
  =#
  #return mu
  IsoMu.cpu!(mu)

  println("computing reactive path")
  ids = @time save_reactive_path(mu, sigma=sigma, out="out/paper/react.pdb", method=path)
  println("saved reactive path")

  p1 = scatter(
    1:size(mu.iso.data[1], 2),
    mu.iso.model(mu.iso.data[1]) |> vec,
    xlabel="frame",
    ylabel="\\chi",
    legend=false,
    markersize=1,
    xticks=0:1000:10000,
    markerstrokewidth=0,
  ) |> display
  savefig("out/paper/paperplot.pdf")

  chi = mu.iso.model(mu.iso.data[1])

  p2 = plot!(ids, chi[ids], linealpha=1, linecolor="Salmon") |> display
  savefig("out/paper/paperplot2.pdf")

  p3 = plot_training(mu.iso)
  savefig("out/paper/learning.pdf")
  println("saved plots")

  rate = ISOKANN.chi_exit_rate(mu.iso, 1)
  println("exit rate: $rate / frame")

  jldsave("out/paper/results.jld2"; mu, ids, p1, p2, p3, rate)
  println("saved jld")
  return (; mu, ids, p1, p2, p3)
end

function ParameterSchedulers._get_opt(scheduler::Scheduler{<:NamedTuple}, t)
  @show kwargs = NamedTuple{keys(scheduler.schedules)}(s(t) for s in scheduler.schedules)
  return scheduler.constructor(; kwargs...)
end