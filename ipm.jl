### log barrier method

using HigherOrderDerivatives
using LinearAlgebra

A = rand(5,5)
b = rand(5)


function barrier(A, x, b, e)
    f = ones(length(b))'*x

    p = 0
    for i in (b - A*x)
        if i > 0
            p += -log(i)
        else
            p += Inf
        end
    end
    return f + e * p
end
    #- e * sum(log.(A*x - b))




function optimize(A, b, epochs)
    e = 0.1
    x = ones(size(A, 2)) #rand(size(A, 2))
    loss = []
    for i in 1:epochs
        tt = t -> barrier(A, t, b, e)
        grad(xx) = gradient(tt, xx)
        hess(xx) = hessian(tt, xx)

        x = x - grad(x) # ((hess(x)+I(size(A, 2))*0.1) \ grad(x))
        e = (1 - 1/(6*sqrt(length(b)))) * e
        append!(loss, sum(b - A*x))
        #append!(loss, barrier(A, x, b, e))
    end
    return loss, x, e
end

using Plots


AA = ones(5,5) .+ rand(5,5)
bb = ones(5) .+ rand(5)

l, w, ee = optimize(AA, bb, 1000)
plot(l)



barrier(AA, w, bb, ee)

sum(bb - AA*w)




Base.isless(z::Dual,w::Dual) = z.f < w.f
Base.isless(z::Real,w::Dual) = z < w.f
Base.isless(z::Dual,w::Real) = z.f < w
