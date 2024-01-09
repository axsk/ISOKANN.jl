function data_vs_loss(iso, log, data)
  ndata = size(iso.data[2], 2) * size(iso.data[2], 3)
  nd = []
  ls = []
  for i in eachindex(log)
    push!(nd, i / length(log) * ndata)
    push!(ls, loss(log[i], data))
  end
  return nd, ls
end

function data_vs_testloss(iso, log, data)
  ndata = size(iso.data[2], 2) * size(iso.data[2], 3)
  is = []
  ls = []
  for i in eachindex(log)
    push!(is, i / length(log) * ndata)
    push!(ls, loss(log[i], data))
  end
  return is, ls
end

function plot_iter_vs_testloss(iso, log, data)
  n = length(iso.losses)
  tl = map(log) do m
    loss(m, data)
  end
  plot!(iso.losses, label="trainloss", yaxis=:log)
  plot!(range(1, n, length(log)), tl, label="testloss")
end
