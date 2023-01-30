######################################################################
##### Replication file for Kaplan and Violante (2010, AEJ:Macro) #####
######################################################################
using Parameters, Statistics, Random, Distributions


##### 1. Housekeeping

@with_kw struct Params
    β::Float64 = 0.975 # Time-discount factor
    γ::Float64 = 2.0 # CRRA
    r::Float64 = 0.04 # Real interest rate
    κ::Array{Float64, 1} = 
    T::Int64 = 35 # Age
    σ_0::Float64 = 0.15 # Volatility of initial permanent income
    σ_η::Float64 = 0.01 # Volatility of permanent income shock
    σ_ϵ::Float64 = 0.05 # Volatility of transitory income shock
    ρ::Float64 = 0.97 # Persistence of permanent income

    m::Float64 = 3.0 # Tauchen maximum Value
    N_A::Int64 = 21 # Grid size for asset 
    N_perp::Int64 = 39 # Grid size for permanent income
    N_trans::Int64 = 19 # Grid size for transitory income
end

mutable struct Results
    bc::Int64 # 1: Zero borrowing  / 2: Natural borrowing limit
    As::Array{Float64, 1} # Asset grid depending on the borrowing limit

    V::Array{Float64, 4} # Value function
    A::Array{Float64, 4} # Savings policy function
    Π_perp::Array{Float64, 2} # Transition probability matrix
    Π_trans::Array{Float64, 2} # Transition probability matrix
    zs::Array{Float64, 1} # States of permanent components
    es::Array{Float64, 1} # States of transitory components
end

function Initialize(bc = 1)
    pars = Params()

    As::Array{Float64, 1} = zeros(pars.N_A)

    V::Array{Float64, 4} = zeros(pars.N_A, pars.T, pars.N_perp, pars.N_trans)
    A::Array{Float64, 4} = zeros(pars.N_A, pars.T, pars.N_perp, pars.N_trans)
    Π_perp::Array{Float64, 2} = zeros(pars.N_perp, pars.N_perp)
    Π_trans::Array{Float64, 2} = zeros(pars.N_trans, pars.N_trans)
    zs::Array{Float64, 1} = zeros(pars.N_perp)
    es::Array{Float64, 1} = zeros(pars.N_trans)

    res = Results(bc, As, V, A, Π_perp, Π_trans, zs, es)
    pars, res
end

##### 1-1. Miscellaneous functions needed

function get_index(val::Float64, grid::Array{Float64,1})
    n = length(grid)
    index = 0
    if val <= grid[1]
        index = 1
    elseif val >= grid[n]
        index = n
    else
        index_upper = findfirst(x->x>val, grid)
        index_lower = index_upper - 1
        val_upper, val_lower = grid[index_upper], grid[index_lower]

        index = index_lower + (val - val_lower) / (val_upper - val_lower)
    end
    return index
end

##### 2. Discretization (Tauchen)

function trans_Tauchen(pars)
    @unpack σ_η, σ_ϵ, ρ, m, N_perp, N_trans = pars

    ## Discretize the space of permanent component

    zN = m*(σ_η^2 / (1-ρ^2))^(1/2)
    z1 = -zN

    Π_perp = zeros(N_perp, N_perp)

    zs = range(start = z1, stop = zN, length = N_perp)
    d1 = zs[2] - zs[1]
    for (j, z_j) in enumerate(zs)
        for (k, z_k) in enumerate(zs)
            if k == 1
                Π_perp[j,k] = cdf(Normal(), (z_k+d1/2-ρ*z_j)/σ_η)
            elseif k == N
                Π_perp[j,k] = 1 - cdf(Normal(), (z_k-d1/2-ρ*z_j)/σ_η)
            end
            Π_perp[j,k] = cdf(Normal(), (z_k+d1/2-ρ*z_j)/σ_η) - cdf(Normal(), (z_k-d1/2-ρ*z_j)/σ_η)
        end
    end

    ## Discretize the space of transitory component

    eN = m*σ_ϵ
    e1 = -eN

    Π_trans = zeros(N_trans, N_trans)

    es = range(start = e1, stop = eN, length = N_trans)
    d2 = es[2] - es[1]
    for (j, e_j) in enumerate(es)
        for (k, e_k) in enumerate(es)
            if k == 1
                Π_trans[j,k] = cdf(Normal(), (e_k+d2/2)/σ_ϵ)
            elseif k == N
                Π_trans[j,k] = 1 - cdf(Normal(), (e_k-d2/2)/σ_ϵ)
            end
            Π_trans[j,k] = cdf(Normal(), (e_k+d/2)/σ_ϵ) - cdf(Normal(), (e_k-d/2)/σ_ϵ)
        end
    end

    return Π_perp, Π_trans, zs, es
end

##### 3. Value function iteration



##### 4. Simulation