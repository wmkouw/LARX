"Utility functions"

function AR_precisionmat(γ, order)
    mW = huge*Matrix{Float64}(I, order, order)
    mW[1, 1] = γ
    return mW
end

function transition(γ, order)
    V = zeros(order, order)
    V[1] = 1/γ
    return V
end

function shift_mat(dim)
    S = Matrix{Float64}(I, dim, dim)
    for i in dim:-1:2
           S[i,:] = S[i-1, :]
    end
    S[1, :] = zeros(dim)
    return S
end

function shift_vec(dim, pos=1)
    s = zeros(dim)
    s[pos] = 1
    return dim == 1 ? s[pos] : s
end

function def_utilmats(order)
    uvector(order), shift(order)
end
