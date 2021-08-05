"Utility functions"

function AR_precisionmat(γ, order)
    mW = 1e8*Matrix{Float64}(I, order, order)
    mW[1,1] = γ
    return mW
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
