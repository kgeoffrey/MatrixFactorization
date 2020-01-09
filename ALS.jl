### ALS method

using StatsBase
using Random
using SparseArrays
using ForwardDiff
using Plots
using LinearAlgebra
using Distributions

function addbias(x::AbstractArray)
    b = hcat(ones(size(x,1)), x)
    return b
end


sparseN(N) = sparse(randperm(N), randperm(N), ones(N), N, N) .* rand(-5:5, N ,N)
mm(N) = sparse(rand(1:N, N), rand(1:N, N), ones(N), N, N)

R =Array(mm(1000))
#R = Array(sparseN(100))
R = abs.(R .* rand(-5:5, 1000 , 1000))


loss(R, X, Y, l) = sum((R - X'*Y).^2) + l*(sum(X.^2) + sum(Y.^2))
function ALS(R, epochs, lam, k)
    lossl = []
    n, m = size(R)

    X = (rand(k, n))
    Y = (rand(k , m))
    L = I(k)*lam
    for _ in 1:epochs
        Y .= ((X*X' + L)\X*R)
        X .= ((Y*Y' + L)\Y*R')

        append!(lossl, loss(R, X, Y, lam))
    end
    return lossl, X, Y
end


R = abs.(R)

@time l, XX, YY = ALS(R, 100, 0.001, 11)
plot(l)


lossi(R, X, Y, l) = sum((R - X*Y').^2) + l*(sum(X.^2) + sum(Y.^2))
function SALS(R, epochs, lam, k)
    lossl = []
    n, m = size(R)
    X = addbias(rand(n, k))
    Y = addbias(rand(m , k))
    L = I(k+1)*lam
    for _ in 1:epochs
        Y .= ((X'*X + L)\X'*R)'
        X .= ((Y'*Y + L)\Y'*R')'

        append!(lossl, lossi(R, X, Y, lam))
    end
    return lossl, X, Y
end

@time l, XX, YY = SALS(R, 100, 0.001, 10)
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

t = rand(10,2)


lossi(R, X, Y, l) = sum((R - X*Y').^2) + l*(sum(X.^2) + sum(Y.^2))

softthres(t, 0.5)

k = 10

n, m = size(t)
X = addbias(rand(n, k))
Y = addbias(rand(m , k))


l1loss(R, X, Y, l) = sum((R - X*Y').^2) + l*(norm(X,1) + norm(Y,1))
function ISTA(R, k, lam, stepsize, epochs)
    lossl = []
    n, m = size(R)
    X = (rand(n, k))
    Y = (rand(m , k))

    for i in 1:epochs
        Y .= softthres(Y + stepsize .* (X'*(R - X*Y'))', lam)
        X .= softthres(X + stepsize .* (Y'*((R - X*Y')'))', lam)
        append!(lossl, l1loss(R, X, Y, lam))
    end
    return lossl, X, Y
end


rr = rand(1000, 100)
lo, xx, yy = ISTA(rr, 50, 0.01, 0.0001, 50)

plot(lo)
RR = (rand(10, 10)*10)
