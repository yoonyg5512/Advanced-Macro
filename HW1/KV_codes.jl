######################################################################
##### Replication file for Kaplan and Violante (2010, AEJ:Macro) #####
######################################################################
using Parameters, Statistics, Random, Distributions


##### 1. Housekeeping

@with_kw struct Params
    β::Float64 = 0.975 # Time-discount factor
    γ::Float64 = 2.0 # CRRA
    r::Float64 = 0.04 # Real interest rate
    T::Float64 = 35.0 # Age
    σ_0::Float64 = 0.15 # Volatility of initial permanent income
    σ_η::Float64 = 0.01 # Volatility of permanent income shock
    σ_ϵ::Float64 = 0.05 # Volatility of transitory income shock
    ρ::Float64 = 0.97 # Persistence of permanent income
end

mutable struct Results
    bc::Int64 # 1: Zero borrowing  / 2: Natural borrowing limit

    A::Array{Float64, } # Savings policy function
end

function Initialize(bc = 1)
    pars = Params()

    
end

##### 2. Discretization (Tauchen)



##### 3. Value function iteration

