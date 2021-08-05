import LinearAlgebra: I, Hermitian, tr, pinv
import ForneyLab: unsafeCov, unsafeMean, unsafePrecision, VariateType

export ruleVariationalLARXOutNPPPPP,
       ruleVariationalLARXIn1PNPPPP,
       ruleVariationalLARXIn2PPNPPP,
       ruleVariationalLARXIn3PPPNPP,
	   ruleVariationalLARXIn4PPPPNP,
	   ruleVariationalLARXIn5PPPPPN

# Autoregression order and bookkeeping matrices
order = Nothing
S = Nothing
s = Nothing

function defineOrder(dim::Int64)
    global order, s, S

	# Set autoregression order
    order = dim

	# Set bookkeeping matrices
    s = shift_vec(order)
    S = shift_mat(order)
end


function ruleVariationalLARXOutNPPPPP(marg_y :: Nothing,
                                      marg_z :: ProbabilityDistribution{Multivariate},
                                      marg_θ :: ProbabilityDistribution{Multivariate},
                                      marg_η :: ProbabilityDistribution{Univariate},
                                      marg_u :: ProbabilityDistribution{Univariate},
                                      marg_γ :: ProbabilityDistribution{Univariate})

    # Expectations of incoming marginal beliefs
    mz = unsafeMean(marg_z)
    mθ = unsafeMean(marg_θ)
    mη = unsafeMean(marg_η)
    mu = unsafeMean(marg_u)
    mγ = unsafeMean(marg_γ)

    # Check order
	if order == Nothing
		defineOrder(length(mz))
	end

		# Construct precision matrix
    Wmγ = AR_precisionmat(mγ, order)

    # Parameters of outgoing message
	Φ = Wmγ
    ϕ = (S + s*mθ')*mz + s*mη*mu

	# Set outgoing message
	return Message(Multivariate, GaussianMeanPrecision, m=ϕ, w=Φ)
end

function ruleVariationalLARXIn1PNPPPP(marg_y :: ProbabilityDistribution{Multivariate},
                                      marg_z :: Nothing,
                                      marg_θ :: ProbabilityDistribution{Multivariate},
                                      marg_η :: ProbabilityDistribution{Univariate},
                                      marg_u :: ProbabilityDistribution{Univariate},
                                      marg_γ :: ProbabilityDistribution{Univariate})

    # Expectations of marginal beliefs
    my = unsafeMean(marg_y)
    mθ = unsafeMean(marg_θ)
    Vθ = unsafeCov(marg_θ)
    mη = unsafeMean(marg_η)
    mu = unsafeMean(marg_u)
    mγ = unsafeMean(marg_γ)

	# Check order
	if order == Nothing
		defineOrder(length(mz))
	end

    # Construct precision matrix
	mW = AR_precisionmat(mγ, order)
	
	# Parameters of outgoing message
    Φ = S'*mW*S + Jx*s'*mW*s*Jx'
    ϕ = (S + s*Jx')'*mW*(my - s*mη*mu) - Jx*(g(mθ,approxx) - Jx'*approxx)*mγ

    return Message(Multivariate, GaussianWeightedMeanPrecision, xi=ϕ, w=Φ)
end

function ruleVariationalLARXIn2PPNPPP(marg_y :: ProbabilityDistribution{Multivariate},
                                      marg_z :: ProbabilityDistribution{Multivariate},
                                      marg_θ :: Nothing,
							  	      marg_η :: ProbabilityDistribution{Univariate},
								      marg_u :: ProbabilityDistribution{Univariate},
                                      marg_γ :: ProbabilityDistribution{Univariate})


	# Expectations of marginal beliefs
	my = unsafeMean(marg_y)
	mz = unsafeMean(marg_z)
	Vz = unsafeCov(marg_z)
	mη = unsafeMean(marg_η)
	mu = unsafeMean(marg_u)
	mγ = unsafeMean(marg_γ)

	# Check order
	if order == Nothing
		defineOrder(length(mz))
	end

    # Construct precision matrix
	mW = AR_precisionmat(mγ, order)

    # Parameters of outgoing message
    Φ = mγ*Jθ*Jθ'
    ϕ = Jθ*s'*mW*(my - s*mη*mu - s*(g(approxθ, mz) - Jθ'*approxθ))

    return Message(Multivariate, GaussianWeightedMeanPrecision, xi=ϕ, w=Φ)
end

function ruleVariationalLARXIn3PPPNPP(marg_y :: ProbabilityDistribution{Multivariate},
                                      marg_z :: ProbabilityDistribution{Multivariate},
                                      marg_θ :: ProbabilityDistribution{Multivariate},
								      marg_η :: Nothing,
								      marg_u :: ProbabilityDistribution{Univariate},
                                      marg_γ :: ProbabilityDistribution{Univariate})

 	# Expectations of marginal beliefs
	mθ = unsafeMean(marg_θ)
    my = unsafeMean(marg_y)
    mz = unsafeMean(marg_z)
	mu = unsafeMean(marg_u)
	vu = unsafeCov(marg_u)
	mγ = unsafeMean(marg_γ)

	# Check order
	if order == Nothing
		defineOrder(length(mz))
	end

    # Map transition noise to matrix
    mW = AR_precisionmat(mγ, order)

	# Parameters of outgoing message
	Φ = mγ*(mu^2 + vu)
    ϕ = (mu*s')*mW*(my - (S*mz + s*g(mθ, mz)))

	return Message(Univariate, GaussianWeightedMeanPrecision, xi=ϕ, w=Φ)
end

function ruleVariationalLARXIn4PPPPNP(marg_y :: ProbabilityDistribution{Multivariate},
                                      marg_z :: ProbabilityDistribution{Multivariate},
                                      marg_θ :: ProbabilityDistribution{Multivariate},
							  	      marg_η :: ProbabilityDistribution{Univariate},
								       marg_u :: Nothing,
                                      marg_γ :: ProbabilityDistribution{Univariate})

 	# Expectations of marginal beliefs
	mθ = unsafeMean(marg_θ)
    my = unsafeMean(marg_y)
    mz = unsafeMean(marg_z)
	mη = unsafeMean(marg_η)
	vη = unsafeCov(marg_η)
	mγ = unsafeMean(marg_γ)

	# Check order
	if order == Nothing
		defineOrder(length(mz))
	end

    # Map transition noise to matrix
    mW = AR_precisionmat(mγ, order)

	# Parameters of outgoing message
	Φ = mγ*(mη^2 + vη)
    ϕ = (mη*s')*mW*(my - (S*mz + s*g(mθ, mz)))

	return Message(Univariate, GaussianWeightedMeanPrecision, xi=ϕ, w=Φ)
end

function ruleVariationalLARXIn5PPPPPN(marg_y :: ProbabilityDistribution{Multivariate},
                                      marg_z :: ProbabilityDistribution{Multivariate},
                                      marg_θ :: ProbabilityDistribution{Multivariate},
							  	      marg_η :: ProbabilityDistribution{Univariate},
								      marg_u :: ProbabilityDistribution{Univariate},
                                      marg_γ :: Nothing)

    # Expectations of marginal beliefs
	mθ = unsafeMean(marg_θ)
    my = unsafeMean(marg_y)
    mz = unsafeMean(marg_z)
	mη = unsafeMean(marg_η)
	mu = unsafeMean(marg_u)
    Vθ = unsafeCov(marg_θ)
    Vy = unsafeCov(marg_y)
    Vz = unsafeCov(marg_z)
	vη = unsafeCov(marg_η)
	vu = unsafeCov(marg_u)

	# Check order
	if order == Nothing
		defineOrder(length(mz))
	end

	# Convenience variables
	Aθx = S*mz + s*g(mθ, mz)

	# Intermediate terms
	term1 = (my*my' + Vy)[1,1]
	term2 = -(Aθx*my')[1,1]
	term3 = -((s*mη*mu)*my')[1,1]
	term4 = -(my*Aθx')[1,1]
	term5 = (mz'*S'*S*mz)[1,1] + (S*Vx*S')[1,1] + g(mθ, mz)^2 + Jx'*Vx*Jx + Jθ'*Vθ*Jθ
	term6 = ((s*mη*mu)*Aθx')[1,1]
	term7 = -(my*(s*mη*mu)')[1,1]
	term8 = (Aθx*(s*mη*mu)')[1,1]
	term9 = (mu^2 + vu)*(mη^2 + vη)

	# Parameters of outgoing message
	a = 3/2
    B = (term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8 + term9)/2

	return Message(Gamma, a=a, b=B)
end
