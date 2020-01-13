### test somthing

using ForwardDiff
using Plots

X = rand(50, 2)
Y = rand(50)

scatter(X,Y)


weird(X, a, b, w) = sum((w .* b - X*a).^2)


function double(X, b, epochs, stepsize)
    a = rand(size(X,2))
    # w = rand(size(b,1))
    w = rand(1)
    loss = []
    te = x -> weird(X,x,b,w)
    ti = x -> weird(X,a,b,x)
    gradl(x) = ForwardDiff.gradient(te, x)
    hessl(x) = ForwardDiff.hessian(te, x)
    graderl(x) = ForwardDiff.gradient(ti, x)
    hesserl(x) = ForwardDiff.hessian(ti, x)
    for i in 1:epochs
        #a = a - stepsize * gradl(a)
        a = a - stepsize * (hessl(a) \ gradl(a))
        #w = w - stepsize * graderl(w)
        w = w - (1-stepsize)* (hesserl(w) \ graderl(w))
        append!(loss, weird(X,a,b,w))
    end
    return loss, a, w
end

loss, a, w = double(X,Y, 20, 0.01)
plot(loss, label = "double")


function single(X, b, epochs, stepsize)
    a = rand(size(X,2))
    loss = []
    meme(X,a,b) = sum((b - X*a).^2)
    te = x -> meme(X,x,b)
    gradl(x) = ForwardDiff.gradient(te, x)
    hess(x) = ForwardDiff.hessian(te, x)
    for i in 1:epochs
        a = a - stepsize* (hess(a) \ gradl(a))
        append!(loss, meme(X,a,b))
    end
    return loss, a
end

lossi, wi = single(X_train, Y_train, 20, .1)
plot!(lossi, label = "single")


######################
using CSV
using Distributions

df = convert(Matrix{Float64}, CSV.read("data.csv", delim = ","))
function splitthis(df, samplesize)
    idx = sample(1:size(df,1), size(df,1))
    l = length(idx)
    df = df[idx, :]
    s = Int(floor(l*samplesize))
    df_train = df[1:l-s, :]
    df_test = df[l-s:end, :]
    X_train = df_train[:,1:end .!= 5]
    Y_train = [if i == 0 (-1) else 1 end for i in df_train[:,5]]
    X_test = df_test[:,1:end .!= 5]
    Y_test = [if i == 0 (-1) else 1 end for i in df_test[:,5]]
    return X_test, Y_test, X_train, Y_train
end
X_test, Y_test, X_train, Y_train = splitthis(df, 0.10)


loss, a, w = double(X_train, Y_train, 20, 0.9)
lossi, wi = single(X_train, Y_train, 20, .9)
plot(loss, label = "double")
plot!(lossi, label = "single")

weird(X_test, a, Y_test, w)

meme(X,a,b) = sum((b - X*a).^2)

meme(X_test, wi, Y_test)

w
