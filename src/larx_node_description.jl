using ForneyLab
using LinearAlgebra
import ForneyLab: SoftFactor, @ensureVariables, generateId, addNode!, associate!,
                  averageEnergy, differentialEntropy, Interface, Variable, slug, ProbabilityDistribution,
				  unsafeLogMean, unsafeMean, unsafeCov, unsafePrecision, unsafeMeanCov,
                  ultimatePartner, region, Region, isClamped, currentInferenceAlgorithm
export LatentAutoregressiveX, LARX

"""
Description:

    A Latent Autoregressive model with eXogenous input (LARX).

    Consider states x ∈ R, with y = [x_k, x_k-1, ... x_k-M+1] and 
    z = [x_k-1, ... x_k-M]. The first element of y (x_k) is produced by an inner 
    product between autoregression coefficients θ and M previous states (i.e. z),
    as well as a coefficient η times an input variable u. The remainder of y is
    produced by the first M-1 elements of z. 

    The node function is a Gaussian with mean-precision parameterization:

    f(y, θ, z, η, u, γ) = 𝒩(y | A(θ)z + B(η)u, V(γ)),

    where A(θ) is a state transition matrix dependent on unknown coefficients θ.
    The matrix is split into a matrix 
        
        S = |0 .. 0
              I   0|, 
        
    which shifts the elements of the state vector downwards and drops the 
    oldest state x_k-M, and a vector s = [1 .. 0]', which can be used to 
    construct a matrix with the autoregression coefficients as top row; 
    
        sθ' = | θ_1 .. θ_M
                 ⋮       ⋮
                 0  ..  0 |
        
    B(η)u is a scaled linear additive control and V(γ) is a covariance 
    matrix based on process precision γ.

Interfaces:

    1. y (state vector at time k; i.e. [x_k, x_k-1, ... x_k-M+1])
    2. θ (autoregression coefficients; i.e. [θ_1, ... θ_M])
    3. z (state vector at time k-1; i.e. [x_k-1, ... x_k-M])
    4. η (control coefficient)
    5. u (control / input)
    6. γ (precision)

Construction:

    LatentAutoregressiveX(y, θ, z, η, u, γ, id=:some_id)
"""


mutable struct LatentAutoregressiveX <: SoftFactor
    id::Symbol
    interfaces::Vector{Interface}
    i::Dict{Symbol,Interface}

    function LatentAutoregressiveX(y, θ, z, η, u, γ; id=generateId(LatentAutoregressiveX))
        @ensureVariables(y, z, θ, η, u, γ)
        self = new(id, Array{Interface}(undef, 6), Dict{Symbol,Interface}())
        addNode!(currentGraph(), self)
        self.i[:y] = self.interfaces[1] = associate!(Interface(self), y)
        self.i[:z] = self.interfaces[2] = associate!(Interface(self), z)
        self.i[:θ] = self.interfaces[3] = associate!(Interface(self), θ)
        self.i[:η] = self.interfaces[4] = associate!(Interface(self), η)
        self.i[:u] = self.interfaces[5] = associate!(Interface(self), u)
        self.i[:γ] = self.interfaces[6] = associate!(Interface(self), γ)
        return self
    end
end

slug(::Type{LatentAutoregressiveX}) = "LARX"

function averageEnergy(::Type{LatentAutoregressiveX},
                       marg_y::ProbabilityDistribution{Multivariate},
                       marg_z::ProbabilityDistribution{Multivariate},
                       marg_θ::ProbabilityDistribution{Multivariate},
                       marg_η::ProbabilityDistribution{Univariate},
                       marg_u::ProbabilityDistribution{Univariate},
                       marg_γ::ProbabilityDistribution{Univariate})

    # Expectations of marginal beliefs
    my, Vy = unsafeMeanCov(marg_y)
    mz, Vz = unsafeMeanCov(marg_z)
    mθ, Vθ = unsafeMeanCov(marg_θ)
    mη, vη = unsafeMeanCov(marg_η)
    mu, vu = unsafeMeanCov(marg_u)
    mγ = unsafeMean(marg_γ)

    error("Not implemented yet")

    # Compute
    Az = mθ*mz
    Eg2 = Eg*Eg' + Jx'*Vz*Jx + Jθ'*Vθ*Jθ

    # Expand square and pre-compute terms
    sq1 = my[1]^2 + Vy[1,1]
	sq2 = my[1]*(Eg + mη*mu)
	sq3 = Eg2 + 2*Eg*mη*mu + (mη^2 + vη)*(mu + vu)

	# Compute average energy
	AE = 1/2*log(2*π) -1/2*unsafeLogMean(marg_γ) +1/2*mγ*(sq1 -2*sq2 + sq3)

    # Correction
    AE += differentialEntropy(marg_y)
    marg_y1 = ProbabilityDistribution(Univariate, GaussianMeanVariance, m=my[1], v=Vy[1,1])
    AE -= differentialEntropy(marg_y1)

    return AE
end

