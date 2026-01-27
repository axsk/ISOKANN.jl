import IterativeSolvers
using SparseArrays, LinearAlgebra

function committor(Q, cl; maxiter=1000)
    A, b = committor_system(Q, cl)
    c = copy(b)

    Pl = Diagonal(A)
    for i in 1:size(Pl, 1)
        Pl[i, i] == 0 && (Pl[i, i] = 1)
    end

    _, hist = IterativeSolvers.gmres!(c, A, b; maxiter=maxiter, Pl=Pl, log=true)
    if !hist.isconverged
        @warn "Committor computation did not converge"
    end

    res = sqrt(sum(abs2, (Pl^-1) * (A * c - b)))
    println("Committor residual mean: ", res)

    return c
end


" solve the committor system where we encode A==1 and B as anything != 0 or 1"
function committor_system(Q, classes)
    #if isnothing(findfirst(x -> x > 0, classes)) ||
    #   isnothing(findfirst(x -> x < 0, classes))
    #    @warn "committor boundary is not well specified"
    #end
    #QQ = copy(Q)
    QQ = sparse(Q') # we work on the transpose since csc column access is fast
    b = copy(classes)
    for i in 1:length(classes)
        if b[i] != 0  # we have a boundary condition
            QQ[:, i] .= 0
            zerocol!(QQ, i)  # note that we work with the transpose
            QQ[i, i] = 1
            if b[i] != 1  # boundary is not 1
                b[i] = 0
            end
        end

        # in the case of Inf outbound rates, ignore this state
        #= alternative to fixinf
        if QQ[i,i] == -Inf
        	zerocol!(QQ, i)
        end
        =#
    end
    QQ = sparse(QQ')
    return QQ, Float64.(b)
end


# set column i of the CSRMatrix Q to 0
# basically the same as `Q[:,i] .= 0`, but way faster
function zerocol!(Q::SparseMatrixCSC, i)
    Q.nzval[Q.colptr[i]:Q.colptr[i+1]-1] .= 0
end

function test_zerocol!(n=10000)
    x = sprand(n, n, 0.01)
    y = copy(x)
    @time y[:, 1] .= 0
    @time zerocol!(x, 1)
    @assert x == y
end