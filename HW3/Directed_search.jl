######################################################################
## Directed search with human capital following Menzio et al. (2012) #
######################################################################
# Prepared by Yeonggyu Yun, Stefano Lord, and Fernando de Lima Lopes #
######################################################################

using Parameters, Statistics, Random, Distributions, Interpolations, Optim, Plots, CSV, Tables

##### 1. Housekeeping

Random.seed!(1234)

@with_kw struct Params
    T::Int64 = 120 # Corresponds to 30 years
    r::Float64 = 0.04 # Interest rate
    β::Float64 = 0.99 # Time discount
    δ::Float64 = 0.1 # Layoff probability
    z::Float64 = 0.4 # Unemployment insurance
    σ::Float64 = 2.0 # Risk aversion
    ξ::Float64 = 1.6 # Matching elasticity
    κ::Float64 = 0.995 # Vacancy cost
    τ::Float64 = 0.2 # Marginal tax rate
    p_L::Float64 = 0.5 # Decrease probability of human capital
    p_H::Float64 = 0.05 # Increase probability of human capital

    # Grids for human capital h and saving choice b and piece-rate ω

    N_h::Int64 = 25
    N_b::Int64 = 101
    N_ω::Int64 = 25
    h_grid::Array{Float64, 1} = range(start = 0.5, stop = 1.5, length = N_h)
    b_grid::Array{Float64, 1} = range(start = 0.0, stop = 10.0, length = N_b)
    ω_grid::Array{Float64, 1} = range(start = 0.0, stop = 1.0, length = N_ω)

    # Simulation

    N_i::Int64 = 1000 # Number of individuals to simulate
end

mutable struct Results
    # Value functions

    U::Array{Float64, 3} # Unemployed
    W::Array{Float64, 4} # Employed
    J::Array{Float64, 3} # Firms

    # Policy functions

    B::Array{Float64, 4} # Asset by unemployment status (and wage), age, human capital, and current net worth
    C::Array{Float64, 4} # Consumption by unemployment status (and wage), age, human capital, and current net worth
    θ::Array{Float64, 3} # Labor market tightness by age, piece rate, and human capital

end

function Initialize()
    pars = Params()

    U::Array{Float64, 3} = zeros(pars.N_b, pars.T, pars.N_h) # Unemployed
    W::Array{Float64, 4} = zeros(pars.N_b, pars.T, pars.N_h, pars.N_ω) # Employed
    J::Array{Float64, 3} = zeros(pars.N_ω, pars.T, pars.N_h) # Firms

    # Policy functions

    B::Array{Float64, 4} = zeros(pars.N_b, pars.T, pars.N_h, 1+pars.N_ω) # Asset by unemployment status (and wage), age, human capital, and current net worth
    C::Array{Float64, 4} = zeros(pars.N_b, pars.T, pars.N_h, 1+pars.N_ω) # Consumption by unemployment status (and wage), age, human capital, and current net worth
    θ::Array{Float64, 3} = zeros(pars.N_ω, pars.T, pars.N_h)# Labor market tightness by age, piece rate, and human capital

    res = Results(U, W, J, B, C, θ)
    return pars, res
end

## 1-1. Miscellaneous functions

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

## Bellman function iteration for a birth cohort

function Bellman(pars, res)
    @unpack T, N_b, N_h, N_ω, β, δ, z, σ, r, ξ, κ, τ, ω_grid, h_grid, b_grid, p_L, p_H = pars
    @unpack U, B, C, J, W, θ = res

    for t in collect(T:-1:1)
        if t == T
            B[:, t, :, :] .= 0
            for (i_ω, ω) in enumerate(ω_grid)
                for (i_h, h) in enumerate(h_grid)
                    J[i_ω, t, i_h] = (1 - ω) * h
                    θ[i_ω, t, i_h] = ((κ / J[i_ω, t, i_h])^(-ξ) - 1)^(1/ξ)
                    
                    for (i_b, b) in enumerate(b_grid)
                        C[i_b, t, i_h, 1] = z + b
                        U[i_b, t, i_h] = (C[i_b, t, i_h, 1]^(1-σ) - 1) / (1-σ)

                        C[i_b, t, i_h, 1+i_ω] = (1 - τ) * ω * h + b
                        W[i_b, t, i_h, i_ω] = (C[i_b, t, i_h, 1+i_ω]^(1-σ) - 1) / (1-σ)
                    end
                end
            end   
        else 
            for (i_ω, ω) in enumerate(ω_grid)
                for (i_h, h) in enumerate(h_grid)
                    J[i_ω, t, i_h] = (1 - ω) * h + β * (p_H * (1 - δ) * J[i_ω, t+1, minimum(i_h + 1, N_h)] + (1-p_H) * (1 - δ) * J[i_ω, t+1, i_h])
                    θ[i_ω, t, i_h] = ((κ / J[i_ω, t, i_h])^(-ξ) - 1)^(1/ξ)
                    
                    for (i_b, b) in enumerate(b_grid)
                        
                        U_cand = zeros(N_b)
                        for (i_pf, pf) in enumerate(b_grid)
                           FB_stay = findmax((θ[:, t+1, i_h].^(-ξ) .+ 1).^(- 1 / ξ) .* W[i_pf, t+1, i_h, 2:(N_ω+1)] .+ (1 - (θ[:, t+1, i_h].^(-ξ) .+ 1).^(- 1 / ξ)) .* U[i_pf, t+1, i_h])[1]
                           FB_decr = findmax((θ[:, t+1, maximum(i_h-1, 1)].^(-ξ) .+ 1).^(- 1 / ξ) .* W[i_pf, t+1, maximum(i_h-1, 1), 2:(N_ω+1)] .+ (1 - (θ[:, t+1, maximum(i_h-1, 1)].^(-ξ) .+ 1).^(- 1 / ξ)) .* U[i_pf, t+1, maximum(i_h-1, 1)])[1]
                           FB = FB_stay * (1-p_L) + FB_decr * p_L

                           U_cand[i_pf] = ((z + b - pf / (1 + r))^(1-σ) - 1) / (1-σ) + β * FB
                        end
                        U[i_b, t, i_h] = findmax(U_cand)[1]
                        B[i_b, t, i_h, 1] = b_grid[findmax(U_cand)[2]]
                        C[i_b, t, i_h, 1] = b_grid[i_b] + z - B[i_b, t, i_h, 1] / (1 + r)

                        W_cand = zeros(N_b)
                        
                        for (i_pf, pf) in enumerate(b_grid)

                        end
                        
                        C[i_b, t, i_h, 1+i_ω] = (1 - τ) * ω * h + b
                        W[i_b, t, i_h, i_ω] = (C[i_b, t, i_h, 1+i_ω]^(1-σ) - 1) / (1-σ)
                    end
                end
            end       
        end
    end
end

## Simulate data with multiple cohorts