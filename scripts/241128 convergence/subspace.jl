"""
Idea: multiple runs can lead to different chis from the dominant subspace.
Test for this looking at the space (e.g. via SVD) spanned by different models trained on the same data
"""

function subspaces(data, n=3, iter=1000)
    isos = [Iso(data) for i in 1:n]
    chis = mapreduce(hcat, isos) do iso
        run!(iso, iter)
        ISOKANN.chi(iso)
    end
    return (; isos, chis)
end

function subspacecomponent(dat)
    res = subspaces(data)
    svd(res.chis)
end