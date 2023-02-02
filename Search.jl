######################################################################
##### Life-cycle Model with random search and human capital      #####
######################################################################
# Prepared by Yeonggyu Yun, Stefano Lord, and Fernando de Lima Lopes #
######################################################################

using Parameters, Statistics, Random, Distributions, Interpolations, Optim, Plots, CSV, Tables

## We used zero borrowing limit for simplicity.

##### 1. Housekeeping

@with_kw struct Params()
    T::Int64 = 360 
    r::Float64 = 0.04 # Interest rate
    β::Float64 = (1/(1+r))^(1/12) # Time discount
    δ::Float64 = 0.033 # Layoff probability
    b::Float64 = 0.1 # Unemployment insurance
    ψ_u::Float64 = 0.4 # Decrease probability of human capital
    ψ_e::Float64 = 0.6 # Increase probability of human capital

    N_h::Int64 = 25
    N_s::Int64 = 41
    h_grid::Array{Float64, 1} = range(start = 1.0, stop = 2.0, length = N_h)
    s_grid::Array{Float64, 1} = range(start = 0.0, stop = 1.0, length = N_s)

    σ_w::Float64 = sqrt(0.1) # Volatility of wage offer
    μ_w::Float64 = 0.5 # Mean of wage offer
    m::Float64 = 3.0 # Tauchen maximum value
    N_w::Int64 = 41 # Grid size for wage offers
end

mutable struct Results()
    U::Array{Float64, 2}
    S::Array{Float64, 2}
    W::Array{Float64, 3}

    ws::Array{Float64, 1}
    Π_w::Array{Float64, 1}
end

function Initialize()
    pars = Params()

    U = zeros(pars.N_h, pars.T)
    S = zeros(pars.N_h, pars.T)
    W = zeros(pars.N_h, pars.T, pars.N_w)
    
    ws = zeros(pars.N_w)
    Π_w = zeros(pars.N_w)
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

function trans_Tauchen(pars)
    @unpack σ_w, μ_w, m, N_w = pars

    wN = m*σ_w
    w1 = -wN

    Π_w = zeros(N_w)

    ws = range(start = w1, stop = wN, length = N_w)
    d = ws[2] - ws[1]
    for (j, w_j) in enumerate(ws)
            if k == 1
                Π_w[j] = cdf(Normal(0, 1), (w_j+d/2)/σ_w)
            elseif k == N_perp
                Π_w[j] = 1 - cdf(Normal(0, 1), (w_j-d/2)/σ_w)
            else
            Π_w[j] = cdf(Normal(0, 1), (w_j+d/2)/σ_w) - cdf(Normal(0, 1), (w_j-d/2)/σ_w)
            end
    end

    return Π_w, ws
end

##### 2. Value function iteration

function Bellman(pars, res)
    @unpack b, β, s_grid, h_grid, T, N_w, N_h, N_s = pars

    res.Π_w, res.ws = trans_Tauchen(pars)

    U_cand = zeros(N_h, T)
    W_cand = zeros(N_h, T, N_w)

    
end