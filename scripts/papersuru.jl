using ISOKANN
using ISOKANN.IsoMu
using Random

using Plots, JLD2

# TODO: update to current data dir
dataprefix = "/data/numerik/ag_cmd/trajectory_transfers_for_isokann/data/Feb19_data_for_paper_2024/des_prod_7UL4_159_251_disulfide_100ns_extension_replica__"

function papersuru(n=30_000; dataprefix=dataprefix)
  Random.seed!(42) # TODO: unfortunately not working - fix it

  links = [DataLink("$dataprefix$i") for i in 1:10]

  mu = isokann(links, learnrate=1e-4, regularization=1e-2)

  # TODO: adjust training schedule?
  train!(mu, n)
  IsoMu.cpu!(mu)

  ids = save_reactive_path(mu, sigma=0.7, out="out/paper/react.pdb")

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

  p3 = plot_learning(mu.iso)
  savefig("out/paper/learning.pdf")

  jldsave("out/paper/results.jld2"; mu, ids, p1, p2, p3)
  return (; mu, ids, p1, p2, p3)
end
