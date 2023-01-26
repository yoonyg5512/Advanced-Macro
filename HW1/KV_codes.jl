######################################################################
##### Replication file for Kaplan and Violante (2010, AEJ:Macro) #####
######################################################################
using Parameters, Statistics, Random, Distributions


##### 1. Housekeeping

@with_kw struct Params
    β::Float64 = 0.975 # Time-discount factor
    γ::Float64 = 2.0 # CRRA
    r::Float64 = 0.04 # Real interest rate
    T::Int64 = 35 # Age
    σ_0::Float64 = 0.15 # Volatility of initial permanent income
    σ_η::Float64 = 0.01 # Volatility of permanent income shock
    σ_ϵ::Float64 = 0.05 # Volatility of transitory income shock
    ρ::Float64 = 0.97 # Persistence of permanent income

    m::Float64 = 3.0 # Tauchen maximum Value
    N_perp::Int64 = 39 # Grid size for permanent income
    N_trans::Int64 = 19 # Grid size for transitory income
end

@with_kw struct Grids


end

mutable struct Results
    bc::Int64 # 1: Zero borrowing  / 2: Natural borrowing limit

    A::Array{Float64, 4} # Savings policy function
    Π::Array{Float64, 2} # Transition probability matrix
end

function Initialize(bc = 1)
    pars = Params()

    A::Array{Float64, 4}

    res = Results(bc, A, Π)
    pars, grids, res
end

##### 2. Discretization (Tauchen)

function trans_Tauchen(pars)

    ## Discretize the space of permanent component

    @unpack σ_η, σ_ϵ, ρ, m, N_perp = pars

    zN = m*(σ_η^2 / (1-ρ^2))^(1/2)
    z1 = -zN

    Π_cand = zeros(N, N)

    zs = range(start = z1, stop = zN, length = N)
    for (j, z_j) in enumerate(zs)
        for (k, z_k) in enumerate(zs)
            if k == 1
                
            elseif k == N

            end
        end
    end

    ## Discretize the space of transitory component
end


##### 3. Value function iteration

##### 4. Simulation