# The von-Mises Fisher distribution (multivariate)
module vMF

module vMFmodel # submodule for component family definitions

export Theta, Data, log_likelihood, log_prior, prior_sample!, new_theta, 
Theta_clear!, Theta_adjoin!, Theta_remove!, 
Hyperparameters, construct_hyperparameters,
update_hyperparameters!, update_parameter!

#=include("Random.jl")
using .Random
=#
using Distributions

typealias Data Array{Float64,1}

type Theta
    mu::Array{Float64,1}    # means
    kappa::Array{Float64,1} # kappa (concentration)
    d::Int64                # dimension
    n::Int64                # number of data points assigned to this cluster
    sum_x::Array{Float64,1} # sum of the data points x assigned to this cluster
    sum_xx::Array{Float64,1}# sum of x.*x for the data points assigned to this cluster
    Theta(mu,kappa) = (p=new(); p.mu=mu; p.kappa=kappa; p.d=length(mu); p.n=0;
        p.sum_x=zeros(p.d); p.sum_xx=zeros(p.d); p)
end

new_theta(H) = Theta(zeros(H.d),ones(H.d))
vMFn_logpdf(x,mu,kappa) = (r=0.0; for i=1:length(x); r+=logpdf(VonMisesFisher(mu[i],kappa[i]),x[i]); end; r)

Theta_clear!(p) = (for i=1:p.d; p.sum_x[i] = 0.0; p.sum_xx[i] = 0.0; end; p.n = 0)
Theta_adjoin!(p,x) = (for i=1:p.d; p.sum_x[i] += x[i]; p.sum_xx[i] += x[i]*x[i]; end; p.n += 1)
Theta_remove!(p,x) = (for i=1:p.d; p.sum_x[i] -= x[i]; p.sum_xx[i] -= x[i]*x[i]; end; p.n -= 1)

type Hyperparameters
    d::Int64    # dimension
    m_0::Float64  # prior mean of mu's (vMF)
    R_0::Float64 # prior concentrarion  mu's (vMF): kappa*R_0
    Q::Float64  # needed for posterior marginal distribution
    a::Float64  # prior mean of kappa's (Gamma distribution)
    b::Float64  # prior sigma of kappa's (Gamma distribution)
end

function construct_hyperparameters(options)
    x = options.x
    d = length(x[1])
    mu = mean(x)
    m_0 = 0.0*length(mu)
    Q = 1.0/8
    R_0 = 0
    a = 1.0
    b = 1.0
    return Hyperparameters(d,m_0,R_0,Q,a,b)
end

log_likelihood(x,p) = vMFn_logpdf(x,p.mu,p.kappa)

log_prior(p,m,R,a,b) = logpdf(VonMisesFisher(m,p.kappa*R),p.mu) + logpdf(Gamma(a,b),p.kappa)
log_prior(p,H) = log_prior(p,H.m_0,H.R_0,H.a,H.b)
prior_sample!(p,H) = (p.kappa = rand(Gamma(H.a,H.b)); p.mu = rand(VonMisesFisher(H.m_0, p.kappa*H.R_0)))

# Hyperpriors

hyperKappa = Gamma(1, 1)
I05(kappa) = quadgk(x -> (1 / π) * exp(kappa * cos(x)) * cos(0.5 * x), 0, π)[1]
C3(kappa) = sqrt(kappa)/I05(kappa)

function update_parameter!(theta_a,theta_b,H,active,density) # Inference via SIR: http://dx.doi.org/10.1080/03610910500308495
    n,sum_x,sum_xx = theta_b.n,theta_b.sum_x,theta_b.sum_xx
    m_n = R0*m0 + sum_x
    r_n = vecnorm(m_n)
    m_n = m_n / R_n
    SIR(theta_a, theta_b)

    h(mu, kappa, H) = C3(kappa)^(H.c+n) * exp(H.kappa * R_n * dot(m_n, mu))
    g(mu, kappa, H) = pdf(VonMisesFisher(H.mu_c,H.Q_c*kappa*R_n), mu) * pdf(hyperKappa,kappa)

    return (density? logp : NaN)
end

function update_hyperparameters!(H,theta,list,t,x,z)
    #error("update_hyperparameters is not yet implemented.")
end

end # module vMFmodel
using .vMFmodel

# Include generic code
include("generic.jl")

# Include core sampler code
include("coreNonconjugate.jl")

end # module MVNaaN
