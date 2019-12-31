### ALS method

using StatsBase
using Random
using SparseArrays
using ForwardDiff
using Plots
using LinearAlgebra



sparseN(N) = sparse(randperm(N), randperm(N), ones(N), N, N) .* rand(-5:5, N ,N)
mm(N) = sparse(rand(1:N, N), rand(1:N, N), ones(N), N, N)

R =Array(mm(100)) + Array(mm(100)) + Array(mm(100))
#R = Array(sparseN(100))

R = (ones(100, 100) - R) .* rand(-5:5, 100 , 100)

rank(R)

k = 10
n, m = size(R)


X = rand(n, k)
Y = rand(m, k)


loss(R, X, Y, l) = sum((R - X*Y').^2) + l*(norm(X) + norm(Y))

w = x -> loss(R, x, Y, 0.1)
v = x -> loss(R, X, x, 0.1)

grad(x) = ForwardDiff.gradient(w, X)


function ALS(R, epochs, lam, k)
    lossl = []
    n, m = size(R)

    X = rand(n, k)
    Y = rand(m ,k)
    L = I(k)*lam

    for i in 1:epochs
        #w = x -> loss(R, x, Y)
        #grad(x) = ForwardDiff.gradient(w, x)
        #X = descent(X, grad, epochs, stepsize)
        #v = y -> loss(R, X, y, 0.1)
        #gradi(x) = ForwardDiff.gradient(v, x)
        #Y = descent(Y, gradi, epochs, stepsize)


        X = ((Y'*Y + L)\Y'*R)'
        Y = ((X'*X + L)\X'*R)'

        append!(lossl, loss(R, X, Y, lam))
    end
    return lossl, X, Y
end



function descent(x, grad, epochs, stepsize)
    for i in 1:epochs
        x = x - stepsize*grad(x)
    end
    return x
end


l, XX, YY = ALS(R, 100, 0.1, 200)
plot(l)


loss(R, XX, YY, 0.0)

l


(XX*YY')

mm(N) = sparse(rand(1:N, N), rand(1:N, N), ones(N), N, N)

rank(Array(mm(10)))
