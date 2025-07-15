using KernelDensity: kde, pdf

function kde_mi(x::AbstractVector, y::AbstractVector; gridsize=100, kdeargs=(;))
    xg = range(extrema(x)...; length=gridsize)
    yg = range(extrema(y)...; length=gridsize)
    dx, dy = step(xg), step(yg)


    pxy = pdf(kde((x, y); kdeargs...), xg, yg)
    px = sum(pxy, dims=2) * dy
    py = sum(pxy, dims=1) * dx


    zlog(x) = x > 0 ? log(x) : 0
    integrand = @. pxy * zlog(pxy / (px * py))
    
    return sum(integrand) * dx * dy
end

function mutual_information(iso; kwargs...)
    f = features(iso) |> cpu
    c = chis(iso) |> vec |> cpu

    mi = [kde_mi(c, f; kwargs...) for f in eachrow(f)]
end