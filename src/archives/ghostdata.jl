""" generate isokann trainings/koopman data with SELF references on the boundary points
this aligns the data points (indices) with the trajectory data, but might introduce a bias"""
function data_from_trajectory_ghosted(xs::Matrix; reverse=false, ghost=true)

  if reverse
    ys = Array{eltype(xs)}(undef, size(xs)..., 2)
    @views ys[:, 1:end-1, 1] .= xs[:, 2:end]  # forward
    @views ys[:, 2:end, 2] .= xs[:, 1:end-1]  # backward

    @views ys[:, end, 1] = ys[:, end, 2]  # twice backward
    @views ys[:, 1, 2] = ys[:, 1, 1]  #twice forward
  else
    ys = Array{eltype(xs)}(undef, size(xs)..., 1)
    @views ys[:, 1:end-1] .= xs[:, 2:end]
    @views ys[:, end] .= xs[:, end]  # self-reference results in 0 change to koopman
  end

  return xs, ys
end