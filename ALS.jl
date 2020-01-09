### ALS method
using StatsBase
using Random
using SparseArrays
using ForwardDiff
using Plots
using LinearAlgebra
using Distributions


## try generate a sparse matrix
sparseN(N) = sparse(randperm(N), randperm(N), ones(N), N, N) .* rand(-5:5, N ,N)
mm(N) = sparse(rand(1:N, N), rand(1:N, N), ones(N), N, N)
R =Array(mm(1000))
#R = Array(sparseN(100))
R = abs.(R .* rand(-5:5, 1000 , 1000))


# loss(R, X, Y, l) = sum((R - X'*Y).^2) + l*(sum(X.^2) + sum(Y.^2))
# function ALS(R, epochs, lam, k)
#     lossl = []
#     n, m = size(R)
#
#     X = (rand(k, n))
#     Y = (rand(k , m))
#     L = I(k)*lam
#     for _ in 1:epochs
#         Y .= ((X*X' + L)\X*R)
#         X .= ((Y*Y' + L)\Y*R')
#
#         append!(lossl, loss(R, X, Y, lam))
#     end
#     return lossl, X, Y
# end
#
#
# R = abs.(R)
#
# @time l, XX, YY = ALS(R, 100, 0.001, 11)
# plot(l)


l2_loss(R, X, Y, l) = sum((R - X*Y').^2) + l*(sum(X.^2) + sum(Y.^2))
function ALS(R, epochs, lam, k)
    lossl = []
    n, m = size(R)
    X = addbias(rand(n, k))
    Y = addbias(rand(m , k))
    L = I(k+1)*lam
    for _ in 1:epochs
        Y .= ((X'*X + L)\X'*R)'
        X .= ((Y'*Y + L)\Y'*R')'

        append!(lossl, l2_loss(R, X, Y, lam))
    end
    return lossl, X, Y
end

@time l, XX, YY = ALS(R, 100, 0.0, 10)
plot(l)


### gradient descent LASSO ###

# soft thresholding thing!

function softthres(x, Y)
    mat = similar(x)
    for i in eachindex(x)
        if x[i] > Y
            mat[i] = x[i] - Y
        elseif x[i] < (-1)*Y
            mat[i] = x[i] + Y
        elseif abs(x[i]) < Y
            mat[i] = 0
        else
            nothing
        end
    end
    return mat
end

function hardthres(x, Y)
    mat = similar(x)
    for i in eachindex(x)
        if abs(x[i]) < Y
            mat[i] = 0
        elseif abs(x[i]) > Y
            mat[i] = x[i]
        else
            nothing
        end
    end
end # Y = sqrt(2*var(X)*log(n)/n))

function sgd(X, Y, n)
    u = sample(1:size(X,1), n, replace = false)
    i = sample(1:size(Y,1), n, replace = false)
    return u, i
end



l1_loss(R, X, Y, l) = sum((R - X*Y').^2) + l*(norm(X,1) + norm(Y,1))

### stochastic Iterative soft thresholding algo for L1 loss
function ISTA(R, k, lam, stepsize, epochs, samplesize)
    lossl = []
    n, m = size(R)
    X = (rand(n, k))
    Y = (rand(m , k))

    for i in 1:epochs
        u, i = sgd(X, Y, samplesize)
        Y[i,:] .= softthres(Y[i,:] + stepsize .* (X[u,:]'*(R[u, i] - X[u,:]*Y[i,:]'))', lam)
        X[u,:] .= softthres(X[u,:] + stepsize .* (Y[i,:]'*((R[u, i] - X[u,:]*Y[i,:]')'))', lam)

        #Y .= softthres(Y + stepsize .* (X'*(R - X*Y'))', lam)
        #X .= softthres(X + stepsize .* (Y'*((R - X*Y')'))', lam)
        append!(lossl, l1_loss(R, X, Y, lam))
    end
    return lossl, X, Y
end

lo, xx, yy = ISTA(R, 100, 0.001, 0.001, 500, 100)

plot(lo)
