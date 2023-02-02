######################################################################
##### Life-cycle Model with random search and human capital      #####
######################################################################
# Prepared by Yeonggyu Yun, Stefano Lord, and Fernando de Lima Lopes #
######################################################################

using Parameters, Statistics, Random, Distributions, Interpolations, Optim, Plots, CSV, Tables

## We used zero borrowing limit for simplicity.

##### 1. Housekeeping

@with_kw struct Params
    T::Int64 = 360 # Corresponds to 30 years
    r::Float64 = 0.04 # Interest rate
    β::Float64 = (1/(1+r))^(1/12) # Time discount
    δ::Float64 = 0.033 # Layoff probability
    b::Float64 = 0.1 # Unemployment insurance
    ψ_u::Float64 = 0.4 # Decrease probability of human capital
    ψ_e::Float64 = 0.6 # Increase probability of human capital

    # Grids for human capital h and search effort s

    N_h::Int64 = 25
    N_s::Int64 = 41
    h_grid::Array{Float64, 1} = range(start = 1.0, stop = 2.0, length = N_h)
    s_grid::Array{Float64, 1} = range(start = 0.0, stop = 1.0, length = N_s)

    # Discretized grid for w

    σ_w::Float64 = sqrt(0.1) # Volatility of wage offer
    μ_w::Float64 = 0.5 # Mean of wage offer
    m::Float64 = 3.0 # Tauchen maximum value
    N_w::Int64 = 41 # Grid size for wage offers
end

mutable struct Results
    U::Array{Float64, 2}
    S::Array{Float64, 2}
    W::Array{Float64, 3}

    w_grid::Array{Float64, 1}
    Π_w::Array{Float64, 1}
end

function Initialize()
    pars = Params()

    U = zeros(pars.N_h, pars.T)
    S = zeros(pars.N_h, pars.T)
    W = zeros(pars.N_h, pars.T, pars.N_w)
    
    w_grid = zeros(pars.N_w)
    Π_w = zeros(pars.N_w)

    res = Results(U, S, W, w_grid, Π_w)
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

function trans_Tauchen(pars)
    @unpack σ_w, μ_w, m, N_w = pars

    wN = m*σ_w
    w1 = -wN

    Π_w = zeros(N_w)

    ws = range(start = w1, stop = wN, length = N_w)
    d = ws[2] - ws[1]
    for (j, w_j) in enumerate(ws)
            if j == 1
                Π_w[j] = cdf(Normal(0, 1), (w_j+d/2)/σ_w)
            elseif j == N_w
                Π_w[j] = 1 - cdf(Normal(0, 1), (w_j-d/2)/σ_w)
            else
            Π_w[j] = cdf(Normal(0, 1), (w_j+d/2)/σ_w) - cdf(Normal(0, 1), (w_j-d/2)/σ_w)
            end
    end

    return Π_w, ws
end

##### 2. Value function iteration

function Bellman(pars, res)
    @unpack b, β, s_grid, h_grid, T, N_w, N_h, N_s, ψ_e, ψ_u, δ = pars
    @unpack Π_w, w_grid, U, W = res

    U_cand = zeros(N_h, T)
    W_cand = zeros(N_h, T, N_w)
    S_cand = zeros(N_h, T)

    for t in collect(T:-1:1)
        if t == T
            U_cand[:,t] .= b
            S_cand[:,t] .= s_grid[1]
            W_cand[:,t,:] .= reshape(h_grid, N_h, 1) * reshape(w_grid, 1, N_w)
        else
            E_W_stay = sum(Π_w .* maximum.(eachcol([W[1,t+1,:]; U[1,t+1]])))
            Us = b .- 0.5 .* s_grid .+ β .* (sqrt.(s_grid) .* E_W_stay .+ (1 .-sqrt.(s_grid)) .* U[1,t+1])

            U_cand[1,t] = findmax(Us)[1]
            S_cand[1,t] = s_grid[findmax(Us)[2]]
            W_cand[1,t,:] = w_grid .* h_grid[1] .+ β .* ψ_e .* ((1-δ) .* W[2,t+1,:] .+ δ .* U[2,t+1]) .+  β .* (1-ψ_e) .* ((1-δ) .* W[1,t+1,:] .+ δ .* U[1,t+1])
            
            E_W_down = sum(Π_w .* maximum.(eachcol([W[N_h-1,t+1,:]; U[N_h-1,t+1]])))
            E_W_stay = sum(Π_w .* maximum.(eachcol([W[N_h,t+1,:]; U[N_h,t+1]])))
            Us = b .- 0.5 .* s_grid .+ β .* ψ_u .* (sqrt.(s_grid) .* E_W_down .+ (1 .-sqrt.(s_grid)) .* U[N_h-1,t+1]) .+ β .* (1-ψ_u) .* (sqrt.(s_grid) .* E_W_stay .+ (1 .-sqrt.(s_grid)) .* U[N_h,t+1])
            
            U_cand[N_h,t] = findmax(Us)[1]
            S_cand[N_h,t] = s_grid[findmax(Us)[2]]
            W_cand[N_h,t,:] = w_grid .* h_grid[N_h] .+ β .* ((1-δ) .* W[N_h,t+1,:] .+ δ .* U[N_h,t+1])
            
            for i in 2:(N_h-1)
                E_W_down = sum(Π_w .* maximum.(eachcol([W[i-1,t+1,:]; U[i-1,t+1]])))
                E_W_stay = sum(Π_w .* maximum.(eachcol([W[i,t+1,:]; U[i,t+1]])))

                Us = b .- 0.5 .* s_grid .+ β .* ψ_u .* (sqrt.(s_grid) .* E_W_down .+ (1 .-sqrt.(s_grid)) .* U[i-1,t+1]) .+ β .* (1-ψ_u) .* (sqrt.(s_grid) .* E_W_stay .+ (1 .-sqrt.(s_grid)) .* U[i,t+1])
                U_cand[i,t] = findmax(Us)[1]
                S_cand[i,t] = s_grid[findmax(Us)[2]]
                
                W_cand[i,t,:] = w_grid .* h_grid[i] .+ β .* ψ_e .* ((1-δ) .* W[i+1,t+1,:] .+ δ .* U[i+1,t+1]) .+  β .* (1-ψ_e) .* ((1-δ) .* W[i,t+1,:] .+ δ .* U[i,t+1])
            end
        end
    end
    
    res.S = S_cand
    return U_cand, W_cand
end

function VFI(pars, res, tol = 1e-4)
    res.Π_w, res.w_grid = trans_Tauchen(pars)

    err = 100
    while err > tol
        U_next, W_next = Bellman(pars, res)

        err = maximum([maximum(abs.(res.U .- U_next)), maximum(abs.(res.W .- W_next))])

        res.U = U_next
        res.W = W_next
    end
end

##### 3. Run

pars, res = Initialize()
VFI(pars, res)