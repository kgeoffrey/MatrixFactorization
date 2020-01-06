### ALS method

using StatsBase
using Random
using SparseArrays
using ForwardDiff
using Plots
using LinearAlgebra
using Distributions



sparseN(N) = sparse(randperm(N), randperm(N), ones(N), N, N) .* rand(-5:5, N ,N)
mm(N) = sparse(rand(1:N, N), rand(1:N, N), ones(N), N, N)

R =Array(mm(1000))
#R = Array(sparseN(100))

R = abs.(R .* rand(-5:5, 1000 , 1000))


loss(R, X, Y, l) = sum((R - X'*Y).^2) + l*(sum(X.^2) + sum(Y.^2))

function ALS(R, epochs, lam, k)
    lossl = []
    n, m = size(R)

    X = rand(k, n)
    Y = rand(k , m)
    L = I(k)*lam
    for _ in 1:epochs
        Y .= ((X*X' + L)\X*R)
        X .= ((Y*Y' + L)\Y*R')

        append!(lossl, loss(R, X, Y, lam))
    end
    return lossl, X, Y
end

function SALS(R, epochs, lam, k)
    lossl = []

    for i in 1:epochs
        RR = R[sample(1:size(R,1), 10, replace = false),:]
        n, m = size(RR)
        X = rand(k, n)
        Y = rand(k , m)
        L = I(k)*lam
        for i in 1:epochs
            Y = ((X*X' + L)\X*RR)
            X = ((Y*Y' + L)\Y*RR')

            append!(lossl, loss(RR, X, Y, lam))
        end
    end
    return lossl, X, Y
end

R = abs.(R)

@time l, XX, YY = ALS(R, 100, 0.001, 10)


plot(l)




(XX'*YY)

(R)
