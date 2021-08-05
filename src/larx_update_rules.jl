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
    mθ,Vθ = unsafeMeanCov(marg_θ)
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
    Φ = S'*Wmγ*S + mγ*(mθ*mθ'+ Vθ)
    ϕ = (S + s*mθ')'*Wmγ*(my - s*mη*mu)

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
	mz,Vz = unsafeMeanCov(marg_z)
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
    Φ = mγ*(mz*mz' + Vz)
    ϕ = mz*s'*Wmγ*(my - s*mη*mu)

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
	mu,vu = unsafeMeanCov(marg_u)
	mγ = unsafeMean(marg_γ)

	# Check order
	if order == Nothing
		defineOrder(length(mz))
	end

    # Construct precision matrix
    Wmγ = AR_precisionmat(mγ, order)

	# Parameters of outgoing message
	Φ = mγ*(mu^2 + vu)
    ϕ = mu*s'*Wmγ*(my - (S + s*mθ')*mz)

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
	mη,vη = unsafeMeanCov(marg_η)
	mγ = unsafeMean(marg_γ)

	# Check order
	if order == Nothing
		defineOrder(length(mz))
	end

    # Construct precision matrix
    Wmγ = AR_precisionmat(mγ, order)

	# Parameters of outgoing message
	Φ = mγ*(mη^2 + vη)
    ϕ = mη*s'*Wmγ*(my - (S + s*mθ')*mz)

	return Message(Univariate, GaussianWeightedMeanPrecision, xi=ϕ, w=Φ)
end

function ruleVariationalLARXIn5PPPPPN(marg_y :: ProbabilityDistribution{Multivariate},
                                      marg_z :: ProbabilityDistribution{Multivariate},
                                      marg_θ :: ProbabilityDistribution{Multivariate},
							  	      marg_η :: ProbabilityDistribution{Univariate},
								      marg_u :: ProbabilityDistribution{Univariate},
                                      marg_γ :: Nothing)

    # Expectations of marginal beliefs
	mθ,Vθ = unsafeMeanCov(marg_θ)
    my,Vy = unsafeMeanCov(marg_y)
    mz,Vz = unsafeMeanCov(marg_z)
	mη,vη = unsafeMeanCov(marg_η)
	mu,vu = unsafeMeanCov(marg_u)

	# Check order
	if order == Nothing
		defineOrder(length(mz))
	end

	# Convenience variables
	EA = (S + s*mθ')
	EB = s*mη

	# Intermediate terms
	term1 = (my*my' + Vy)[1,1]
	term2 = ((EA*mz + EB*mu)*my')[1,1]
	term3 = (my*(EA*mz + EB*mu)')[1,1]
	term4 = ((S'*S + S'*s*mθ' + mθ*s'*S + mθ*mθ' + Vθ)*(mz*mz' + Vz))[1,1]
	term5 = (EB*mu*mz'*EA')[1,1]
	term6 = (EA*mz*mu'*EB')[1,1]
	term7 = (mu*s*(mη*mη' + vη)*s')[1,1]

	# Parameters of outgoing message
	a = 3/2
    B = 1/2*(term1 - term2 - term3 + term4 + term5 + term6 + term7)

	return Message(Gamma, a=a, b=B)
end
