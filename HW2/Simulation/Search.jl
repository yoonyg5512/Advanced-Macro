######################################################################
##### Life-cycle Model with random search and human capital      #####
######################################################################
# Prepared by Yeonggyu Yun, Stefano Lord, and Fernando de Lima Lopes #
######################################################################

using Parameters, Statistics, Random, Distributions, Interpolations, Optim, Plots, CSV, Tables

## We used zero borrowing limit for simplicity.

##### 1. Housekeeping

Random.seed!(1234)

@with_kw struct Params
    T::Int64 = 360 # Corresponds to 30 years
    r::Float64 = 0.04 # Interest rate
    β::Float64 = (1/(1+r))^(1/12) # Time discount
    δ::Float64 = 0.033 # Layoff probability
    b::Float64 = 0.1 # Unemployment insurance

    # Grids for human capital h and search effort s

    N_h::Int64 = 25
    N_s::Int64 = 41
    h_grid::Array{Float64, 1} = range(start = 1.0, stop = 2.0, length = N_h)
    s_grid::Array{Float64, 1} = range(start = 0.0, stop = 1.0, length = N_s)

    # Discretized grid for w

    σ_w::Float64 = sqrt(0.1) # Volatility of wage offer
    μ_w::Float64 = 0.5 # Mean of wage offer
    m::Float64 = 3 # Tauchen maximum value
    N_w::Int64 = 41 # Grid size for wage offers

    # Simulation

    N_i::Int64 = 1000 # Number of individuals to simulate
end

mutable struct Results
    ψ_u::Float64
    ψ_e::Float64

    U::Array{Float64, 2}
    S::Array{Float64, 2}
    W::Array{Float64, 3}

    w_grid::Array{Float64, 1}
    Π_w::Array{Float64, 1}
end

function Initialize()
    pars = Params()

    ψ_u::Float64 = 0.236 # Decrease probability of human capital
    ψ_e::Float64 = 0.052# Increase probability of human capital

    U = zeros(pars.N_h, pars.T)
    S = zeros(pars.N_h, pars.T)
    W = zeros(pars.N_h, pars.T, pars.N_w)
    
    w_grid = zeros(pars.N_w)
    Π_w = zeros(pars.N_w)

    res = Results(ψ_u, ψ_e, U, S, W, w_grid, Π_w)
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

    ws = range(start = w1, stop = wN, length = N_w) .+ μ_w
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
    @unpack b, β, s_grid, h_grid, T, N_w, N_h, N_s, δ = pars
    @unpack Π_w, w_grid, U, ψ_e, ψ_u, W = res

    U_cand = zeros(N_h, T)
    W_cand = zeros(N_h, T, N_w)
    S_cand = zeros(N_h, T)

    for t in collect(T:-1:1)
        if t == T
            U_cand[:,t] .= b
            S_cand[:,t] .= s_grid[1]
            W_cand[:,t,:] .= reshape(h_grid, N_h, 1) * reshape(w_grid, 1, N_w)
        else
            # If h is at its lower bound, there's no decrease in human capital.

            E_W_stay = sum(Π_w .* maximum.(eachcol(vcat(reshape(W[1,t+1,:], 1, N_w), reshape(ones(N_w).*U[1,t+1], 1, N_w)))))
            Us = b .- 0.5 .* s_grid .+ β .* (sqrt.(s_grid) .* E_W_stay .+ (1 .-sqrt.(s_grid)) .* U[1,t+1])

            U_cand[1,t] = findmax(Us)[1]
            S_cand[1,t] = s_grid[findmax(Us)[2]]
            W_cand[1,t,:] = w_grid .* h_grid[1] .+ β .* ψ_e .* ((1-δ) .* W[2,t+1,:] .+ δ .* U[2,t+1]) .+  β .* (1-ψ_e) .* ((1-δ) .* W[1,t+1,:] .+ δ .* U[1,t+1])
            
            # If h is at its upper bound, there's no increase in human capital.

            E_W_down = sum(Π_w .* maximum.(eachcol(vcat(reshape(W[N_h-1,t+1,:], 1, N_w), reshape(ones(N_w).*U[N_h-1,t+1], 1, N_w)))))
            E_W_stay = sum(Π_w .* maximum.(eachcol(vcat(reshape(W[N_h,t+1,:], 1, N_w), reshape(ones(N_w).*U[N_h,t+1], 1, N_w)))))
            Us = b .- 0.5 .* s_grid .+ β .* ψ_u .* (sqrt.(s_grid) .* E_W_down .+ (1 .-sqrt.(s_grid)) .* U[N_h-1,t+1]) .+ β .* (1-ψ_u) .* (sqrt.(s_grid) .* E_W_stay .+ (1 .-sqrt.(s_grid)) .* U[N_h,t+1])
            
            U_cand[N_h,t] = findmax(Us)[1]
            S_cand[N_h,t] = s_grid[findmax(Us)[2]]
            W_cand[N_h,t,:] = w_grid .* h_grid[N_h] .+ β .* ((1-δ) .* W[N_h,t+1,:] .+ δ .* U[N_h,t+1])
            
            for i in 2:(N_h-1)
                E_W_down = sum(Π_w .* maximum.(eachcol(vcat(reshape(W[i-1,t+1,:], 1, N_w), reshape(ones(N_w).*U[i-1,t+1], 1, N_w)))))
                E_W_stay = sum(Π_w .* maximum.(eachcol(vcat(reshape(W[i,t+1,:], 1, N_w), reshape(ones(N_w).*U[i,t+1], 1, N_w)))))

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

##### 4. Simulation

mutable struct dat_sim
    H_path::Array{Float64, 2} # Human capital h
    W_path::Array{Float64, 2} # Wages w
    U_path::Array{Float64, 2} # 1 if employed, 0 if unemployed
end

function Init_sim(pars)
    H_path::Array{Float64, 2} = zeros(pars.N_i, pars.T)
    W_path::Array{Float64, 2} = zeros(pars.N_i, pars.T)
    U_path::Array{Float64, 2} = zeros(pars.N_i, pars.T)

    simulated = dat_sim(H_path, W_path, U_path)
    return simulated
end

function simulate(pars, res)
    @unpack b, β, s_grid, h_grid, T, N_w, N_h, N_s, δ, N_i = pars
    @unpack Π_w, w_grid, U, S, W, ψ_e, ψ_u= res

    μ_h = 1.5 # Mean of Uniform(1,2)
    σ_h = sqrt(1/12) # Standard deviation of Uniform(1,2)
   
    H_path::Array{Float64, 2} = zeros(pars.N_i, pars.T)
    W_path::Array{Float64, 2} = zeros(pars.N_i, pars.T)
    U_path::Array{Float64, 2} = zeros(pars.N_i, pars.T)

    H_path[:,1] = rand(Normal(μ_h, σ_h), N_i)
    U_path[:,1] .= 0
    W_path[:,1] .= b
    for i in 1:N_i
        H_path[i,1] = findmax(-abs.(H_path[i,1] .- h_grid))[2] # map initial human capital onto the discretized set of hs
    end
    
    for i in 1:N_i
        for t in 2:T
            prob = rand(Uniform(0,1))

            if U_path[i,t-1] == 0
                pi_s = sqrt(S[Int(H_path[i,t-1]), t-1])

                p_unmatched_same = 0
                p_unmatched_decr = 0
                p_offer_same = zeros(N_w)
                p_offer_decr = zeros(N_w)

                for w_i in 1:N_w
                    p_unmatched_same += pi_s * (1-ψ_u) * (Π_w[w_i] * ifelse(W[Int(H_path[i,t-1]), t, w_i] < U[Int(H_path[i,t-1]), t], 1, 0))
                    p_unmatched_decr += pi_s * ψ_u * (Π_w[w_i] * ifelse(W[maximum([Int(H_path[i,t-1])-1, 1]), t, w_i] < U[maximum([Int(H_path[i,t-1])-1, 1]), t], 1, 0)) 
                    p_offer_decr[w_i] = pi_s * ψ_u * Π_w[w_i] * ifelse(W[maximum([Int(H_path[i,t-1])-1, 1]), t, w_i] >= U[maximum([Int(H_path[i,t-1])-1, 1]), t], 1, 0)
                    p_offer_same[w_i] = pi_s * (1-ψ_u) * Π_w[w_i] * ifelse(W[Int(H_path[i,t-1]), t, w_i] >= U[Int(H_path[i,t-1]), t], 1, 0)                                
                end

                p_unmatched_same = p_unmatched_same + (1-ψ_u) * (1-pi_s)
                p_unmatched_decr = p_unmatched_decr + ψ_u * (1-pi_s)

                p_noncon = [p_unmatched_same; p_offer_same; p_unmatched_decr; p_offer_decr]
                ws_grid = [b; w_grid; b; w_grid]
                p_grid = cumsum(p_noncon)
                new_state = ceil(get_index(prob, p_grid))
                
                U_path[i,t] = ifelse(new_state !== 1 && new_state !== (N_w+2) && p_noncon[Int(new_state)] > 0, 1, 0)
                W_path[i,t] = ws_grid[Int(new_state)] * ifelse(p_noncon[Int(new_state)] > 0, 1, 0)
                H_path[i,t] = maximum([H_path[i,t-1] - 1 * ifelse(new_state > (N_w + 1), 1, 0), 1])

            elseif U_path[i,t-1] == 1
                p_grid = cumsum([(1-δ)*(1-ψ_e), (1-δ)*ψ_e, δ*(1-ψ_e), δ*ψ_e])
                new_state = ceil(get_index(prob, p_grid))

                if new_state == 1 # Employed and Same
                    U_path[i,t] = 1
                    W_path[i,t] = W_path[i,t-1]
                    H_path[i,t] = H_path[i,t-1]
                elseif new_state == 2 # Employed and Higher
                    U_path[i,t] = 1
                    W_path[i,t] = W_path[i,t-1]
                    H_path[i,t] = minimum([H_path[i,t-1]+1, N_h])
                elseif new_state == 3 # Unemployed and Same
                    U_path[i,t] = 0
                    W_path[i,t] = b
                    H_path[i,t] = H_path[i,t-1]
                elseif new_state == 4 # Unemployed and Higher
                    U_path[i,t] = 0
                    W_path[i,t] = b
                    H_path[i,t] = minimum([H_path[i,t-1]+1, N_h])
                end
            end
        end
    end
    return U_path, H_path, W_path
end

##### 6. Get simulated data

simulated = Init_sim(pars)
simulated.U_path, simulated.H_path, simulated.W_path = simulate(pars, res)

