### ALS method
using Random
using SparseArrays
using Plots
using LinearAlgebra
using Distributions


## try generate a sparse matrix
sparseN(N) = sparse(randperm(N), randperm(N), ones(N), N, N) .* rand(-5:5, N ,N)
mm(N) = sparse(rand(1:N, N), rand(1:N, N), ones(N), N, N)
R = Array(mm(1000))
R = abs.(R .* rand(1:5, 1000 , 1000))


rank(R)


l2_loss(R, X, Y, l) = sum((R - X*Y').^2) + l*(sum(X.^2) + sum(Y.^2))

## closed form solution using all samples (full matrix)
function ALS(R, epochs, lam, k)
    loss = []
    n, m = size(R)
    X = (rand(n, k))
    Y = (rand(m , k))
    L = I(k)*lam
    for _ in 1:epochs
        Y .= ((X'*X + L)\X'*R)'
        X .= ((Y'*Y + L)\Y'*R')'
        append!(loss, l2_loss(R, X, Y, lam))
    end
    return loss, X, Y
end

@time l, XX, YY = ALS(R, 500, 0.0, 100)
plot(l, label = "ALS")

## stochastic alternating least squares
function SALS(R, k, lam, stepsize, epochs, samplesize)
    loss = []
    n, m = size(R)
    X = (rand(n, k))
    Y = (rand(m , k))

    for i in 1:epochs
        u, i = sgd(X, Y, samplesize)
        Y[i,:] .+= stepsize .* (X[u,:]'*(R[u, i] - X[u,:]*Y[i,:]'))' + lam .* Y[i,:]
        X[u,:] .+= stepsize .* (Y[i,:]'*((R[u, i] - X[u,:]*Y[i,:]')'))' + lam .* X[u,:]
        append!(loss, l2_loss(R, X, Y, lam))
    end
    return loss, X, Y
end

lo, xx, yy = SALS(R, 100, 0.01, 0.0001, 500, 50)
plot(lo, label = "SALS")

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
    return mat
end # Y = sqrt(2*var(X)*log(n)/n))

function sgd(X, Y, n)
    u = sample(1:size(X,1), n, replace = false)
    i = sample(1:size(Y,1), n, replace = false)
    return u, i
end


l1_loss(R, X, Y, l) = sum((R - X*Y').^2) + l*(norm(X,1) + norm(Y,1))

### stochastic Iterative soft thresholding algo for L1 loss
function ISTA(R, k, lam, stepsize, epochs, samplesize)
    loss = []
    n, m = size(R)
    X = (rand(n, k))
    Y = (rand(m , k))

    for i in 1:epochs
        u, i = sgd(X, Y, samplesize)
        Y[i,:] .= softthres(Y[i,:] + stepsize .* (X[u,:]'*(R[u, i] - X[u,:]*Y[i,:]'))', lam)
        X[u,:] .= softthres(X[u,:] + stepsize .* (Y[i,:]'*((R[u, i] - X[u,:]*Y[i,:]')'))', lam)
        append!(loss, l1_loss(R, X, Y, lam))
    end
    return loss, X, Y
end

lo, xx, yy = ISTA(R, 100, 0.01, 0.0001, 500, 50)
plot!(lo, label = "ISTA")


l0_loss(R, X, Y, l) = sum((R - X*Y').^2) + l*(norm(X,0) + norm(Y,0))
### stochastic Iterative hard thresholding algo for L0 loss
function IHTA(R, k, lam, stepsize, epochs, samplesize)
    loss = []
    n, m = size(R)
    X = (rand(n, k))
    Y = (rand(m , k))

    for i in 1:epochs
        u, i = sgd(X, Y, samplesize)
        Y[i,:] .= hardthres(Y[i,:] + stepsize .* (X[u,:]'*(R[u, i] - X[u,:]*Y[i,:]'))', lam)
        X[u,:] .= hardthres(X[u,:] + stepsize .* (Y[i,:]'*((R[u, i] - X[u,:]*Y[i,:]')'))', lam)
        append!(loss, l0_loss(R, X, Y, lam))
    end
    return loss, X, Y
end

lo, xx, yy = IHTA(R, 100, 0.01, 0.0001, 500, 50)
plot!(lo, label = "IHTA")


using CSV

df = convert(Matrix{Float64}, CSV.read("ml-latest-small/ratings.csv", delim = ","))

function prepare(df)
    users = df[:,1]
    movies = df[:,2]
    #movies = Array(1:length(unique(df[:,2])))
    rating = df[:,3]
    usersl = length(unique(df[:,1]))
    moviesl = length(unique(df[:,2]))

    mat = Array{Float64}(undef, usersl, Int(maximum(df[:,2])))
    mat2 = []
    for (i, v) in enumerate(users)
        mat[Int(v), Int(movies[Int(i)])] = rating[Int(i)]
    end
    return mat
end



tt = prepare(df)
tt

matl = []

for i in 1:size(tt,2)
    if sum(tt[:,i]) > 0
        hcat(tt[:,i], matl)
    end
end
