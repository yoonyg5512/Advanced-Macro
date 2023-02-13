######################################################################
## Directed search with human capital following Menzio et al. (2012) #
######################################################################
# Prepared by Yeonggyu Yun, Stefano Lord, and Fernando de Lima Lopes #
######################################################################

using Parameters, Statistics, Plots, CSV, Tables, Random, Distributions, DataFrames

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
    E::Array{Float64, 4} # Wage choice when unemployed by wealth, age, human capital level, and two occasions of staying and decreasing the human capital

end

function Initialize()
    pars = Params()

    U::Array{Float64, 3} = zeros(pars.N_b, pars.T, pars.N_h) # Unemployed
    W::Array{Float64, 4} = zeros(pars.N_b, pars.T, pars.N_h, pars.N_ω) # Employed
    J::Array{Float64, 3} = zeros(pars.N_ω, pars.T, pars.N_h) # Firms

    # Policy functions

    B::Array{Float64, 4} = zeros(pars.N_b, pars.T, pars.N_h, 1+pars.N_ω) # Asset by unemployment status (and wage), age, human capital, and current net worth
    C::Array{Float64, 4} = zeros(pars.N_b, pars.T, pars.N_h, 1+pars.N_ω) # Consumption by unemployment status (and wage), age, human capital, and current net worth
    θ::Array{Float64, 3} = zeros(pars.N_ω, pars.T, pars.N_h) # Labor market tightness by age, piece rate, and human capital
    E::Array{Float64, 4} = zeros(pars.N_b, pars.T-1, pars.N_h, 2)

    res = Results(U, W, J, B, C, θ, E)
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

## 2. Bellman function iteration for a birth cohort
## Speed is not too bad.

function Bellman(pars, res)
    @unpack T, N_b, N_h, N_ω, β, δ, z, σ, r, ξ, κ, τ, ω_grid, h_grid, b_grid, p_L, p_H = pars

    U = zeros(N_b, T, N_h) # Unemployed
    W = zeros(N_b, T, N_h, N_ω) # Employed
    J = zeros(N_ω, T, N_h) # Firms
    B = zeros(N_b, T, N_h, 1+N_ω) # Asset by unemployment status (and wage), age, human capital, and current net worth
    C = zeros(N_b, T, N_h, 1+N_ω) # Consumption by unemployment status (and wage), age, human capital, and current net worth
    θ = zeros(N_ω, T, N_h) # Labor market tightness by age, piece rate, and human capital
    E = zeros(N_b, T-1, N_h, 2)

    for t in collect(T:-1:1)
        println(t)
        if t == T
            B[:, t, :, :] .= 0
            for (i_b, b) in enumerate(b_grid)
                for (i_h, h) in enumerate(h_grid)

                    C[i_b, t, i_h, 1] = z + b
                    U[i_b, t, i_h] = (C[i_b, t, i_h, 1]^(1-σ) - 1) / (1-σ)

                    for (i_ω, ω) in enumerate(ω_grid)
                
                        J[i_ω, t, i_h] = (1 - ω) * h

                        if κ / J[i_ω, t, i_h] < 1
                            θ[i_ω, t, i_h] = ((κ / J[i_ω, t, i_h])^(-ξ) - 1)^(1/ξ)
                        elseif κ / J[i_ω, t, i_h] >= 1
                            θ[i_ω, t, i_h] = 0
                        end
                        
                        C[i_b, t, i_h, 1+i_ω] = (1 - τ) * ω * h + b
                        W[i_b, t, i_h, i_ω] = (C[i_b, t, i_h, 1+i_ω]^(1-σ) - 1) / (1-σ)
                    end
                end
            end   
        else 
            for (i_b, b) in enumerate(b_grid)
                for (i_h, h) in enumerate(h_grid)
                    U_cand = zeros(N_b)
                    E_cand = zeros(N_b, 2)
                        
                    for (i_pf, pf) in enumerate(b_grid)
                        FB_s = (θ[:, t+1, i_h].^(-ξ) .+ 1).^(- 1 / ξ) .* W[i_pf, t+1, i_h, :] .+ (1 .- (θ[:, t+1, i_h].^(-ξ) .+ 1).^(- 1 / ξ)) .* U[i_pf, t+1, i_h]
                        FB_s[isnan.(FB_s)] = (1 .- (θ[isnan.(FB_s), t+1, i_h].^(-ξ) .+ 1).^(- 1 / ξ)) .* U[i_pf, t+1, i_h]

                        FB_stay, w1 = findmax(FB_s)
                           
                        FB_d = (θ[:, t+1, max(i_h-1, 1)].^(-ξ) .+ 1).^(- 1 / ξ) .* W[i_pf, t+1, max(i_h-1, 1), :] .+ (1 .- (θ[:, t+1, max(i_h-1, 1)].^(-ξ) .+ 1).^(- 1 / ξ)) .* U[i_pf, t+1, max(i_h-1, 1)]
                        FB_d[isnan.(FB_d)] = (1 .- (θ[isnan.(FB_d), t+1, max(i_h-1, 1)].^(-ξ) .+ 1).^(- 1 / ξ)) .* U[i_pf, t+1, max(i_h-1, 1)]

                        FB_decr, w2 = findmax(FB_d)
                           
                        FB = FB_stay * (1-p_L) + FB_decr * p_L

                        U_cand[i_pf] = ifelse(z + b - pf / (1 + r) <= 0, -Inf, ((z + b - pf / (1 + r))^(1-σ) - 1) / (1-σ) + β * FB)
                        E_cand[i_pf,:] = [w1, w2]
                    end
                    pol = findmax(U_cand)

                    U[i_b, t, i_h] = pol[1]
                    E[i_b, t, i_h, 1] = ω_grid[Int(E_cand[pol[2],1])]
                    E[i_b, t, i_h, 2] = ω_grid[Int(E_cand[pol[2],2])]

                    B[i_b, t, i_h, 1] = b_grid[pol[2]]
                    C[i_b, t, i_h, 1] = b_grid[i_b] + z - B[i_b, t, i_h, 1] / (1 + r)

                    for (i_ω, ω) in enumerate(ω_grid)

                        J[i_ω, t, i_h] = (1 - ω) * h + β * (p_H * (1 - δ) * J[i_ω, t+1, min(i_h + 1, N_h)] + (1-p_H) * (1 - δ) * J[i_ω, t+1, i_h])

                        if κ / J[i_ω, t, i_h] < 1
                            θ[i_ω, t, i_h] = ((κ / J[i_ω, t, i_h])^(-ξ) - 1)^(1/ξ)
                        elseif κ / J[i_ω, t, i_h] >= 1
                            θ[i_ω, t, i_h] = 0
                        end
                        
                        W_cand = zeros(N_b)
                        
                        for (i_pf, pf) in enumerate(b_grid)
                            FB_stay = (1-δ) * W[i_pf, t+1, i_h, i_ω] + δ * U[i_pf, t+1, i_h]
                            FB_incr = (1-δ) * W[i_pf, t+1, min(i_h+1, N_h), i_ω] + δ * U[i_pf, t+1, min(i_h+1, N_h)]
                            FB = FB_stay * (1-p_H) + FB_incr * p_H

                            W_cand[i_pf] = ifelse(ω * h * (1-τ) + b - pf / (1 + r) <= 0, -Inf, ((ω * h * (1-τ) + b - pf / (1 + r))^(1-σ) - 1) / (1-σ) + β * FB)
                        end
                        pol_e = findmax(W_cand)
                        W[i_b, t, i_h, i_ω] = pol_e[1]
                        B[i_b, t, i_h, 1 + i_ω] = b_grid[pol_e[2]]
                        C[i_b, t, i_h, 1+i_ω] = (1 - τ) * ω * h + b - B[i_b, t, i_h, 1 + i_ω] / (1+r)
                    end
                end
            end       
        end
    end

    return U, B, C, J, W, θ, E
end

function VFI(pars, res)
    res.U, res.B, res.C, res.J, res.W, res.θ, res.E = Bellman(pars, res)
end

## 3. Simulate data with multiple cohorts

mutable struct Simulated
    B_sim::Array{Float64, 2} # Wealth
    C_sim::Array{Float64, 2} # Consumption
    W_sim::Array{Float64, 2} # After-tax/transfer wage
    U_sim::Array{Int64, 2} # Unemployment stats (1: unemployment, 0: employment)
    T_sim::Array{Int64, 2} # Ages
    H_sim::Array{Float64, 2} # Human capital
    id_sim::Array{Float64, 2} # Individual id (in panel data)
end

function Init_sim(pars)
    B_sim::Array{Float64, 2} = zeros(pars.N_i, 180) 
    C_sim::Array{Float64, 2} = zeros(pars.N_i, 180) 
    W_sim::Array{Float64, 2} = zeros(pars.N_i, 180) 
    U_sim::Array{Int64, 2} = zeros(pars.N_i, 180) 
    T_sim::Array{Int64, 2} = zeros(pars.N_i, 180)
    H_sim::Array{Int64, 2} = zeros(pars.N_i, 180)
    id_sim::Array{Int64, 2} = zeros(pars.N_i, 180)

    sim = Simulated(B_sim, C_sim, W_sim, U_sim, T_sim, H_sim, id_sim)
end

function Simulate(pars, res)
    @unpack δ, p_H, p_L, T, N_i, N_h, N_b, N_ω, z, h_grid, ω_grid, b_grid, τ, r, ξ = pars
    @unpack B, C, θ, E = res

    total::Int64 = 300
    burnin::Int64 = T

    B_total = zeros(N_i, total)
    C_total = zeros(N_i, total)
    W_total = zeros(N_i, total)
    U_total::Array{Int64, 2} = zeros(N_i, total)
    T_total::Array{Int64, 2} = zeros(N_i, total)
    H_total::Array{Int64, 2} = zeros(N_i, total)
    id_total::Array{Int64, 2} = zeros(N_i, total)

    init_ages = rand(1:T, N_i)
    for i in 1:N_i
        if i == 1
            id_total[i,1] = 1
        else
            id_total[i,1] = id_total[i-1, total] + 1
        end
        T_total[i,1] = init_ages[i]
        h_init = ifelse(init_ages[i] == 1, 1, rand(1:N_h))
        u_init = ifelse(init_ages[i] == 1, 1, rand(1:2)-1)
        b_init = ifelse(init_ages[i] == 1, 1, rand(1:N_b))
        w_init = ifelse(u_init == 1, 0, rand(1:N_ω))

        B_total[i,1] = b_grid[b_init]
        W_total[i,1] = ifelse(u_init == 1, z, ω_grid[max(1,w_init)] * h_grid[max(1,h_init)] * (1- τ))
        U_total[i,1] = u_init
        C_total[i,1] = C[b_init, T_total[i,1], h_init, ifelse(u_init == 1, 1, 1+w_init)]
        H_total[i,1] = h_init
        S = B[b_init, T_total[i,1], h_init, ifelse(u_init == 1, 1, 1+w_init)]
        omega = w_init

        for t in 2:total
            prob = rand(Uniform(0,1))

            if T_total[i,t-1] == T
                id_total[i,t] = id_total[i,t-1] + 1
                T_total[i,t] = 1
                B_total[i,t] = 0
                U_total[i,t] = 1
                W_total[i,t] = z
                H_total[i,t] = 1
                C_total[i,t] = C[1,1,1,1]
                S = (W_total[i,t] + B_total[i,t] - C_total[i,t]) * (1+r)
                omega = 0

            elseif T_total[i,t-1] < T
                id_total[i,t] = id_total[i, t-1]
                T_total[i,t] = T_total[i,t-1] + 1
                B_total[i,t] = S
                i_b = round(get_index(B_total[i,t], b_grid))
                
                if U_total[i,t-1] == 1
                    i_b_pre = round(get_index(B_total[i,t-1], b_grid))
                    omega_s = round(get_index(E[Int(i_b_pre), T_total[i,t-1], H_total[i,t-1], 1], ω_grid))
                    omega_d = round(get_index(E[Int(i_b_pre), T_total[i,t-1], max(H_total[i,t-1]-1, 1), 2], ω_grid))
                    θ_find_stay = θ[Int(omega_s), T_total[i,t], H_total[i,t-1]]
                    p_find_stay = (θ_find_stay^(-ξ) + 1)^(-1/ξ)
                    θ_find_decr = θ[Int(omega_d), T_total[i,t], max(H_total[i,t-1]-1, 1)]
                    p_find_decr = (θ_find_decr^(-ξ) + 1)^(-1/ξ)
                    
                    p_grid = [p_L * p_find_decr, p_L * (1-p_find_decr), (1-p_L) * p_find_stay, (1-p_L) * (1-p_find_stay)]
                    p_grid = cumsum(p_grid)
                    p_case = ceil(get_index(prob, p_grid))

                    if p_case == 1
                        U_total[i,t] = 0
                        H_total[i,t] = max(H_total[i,t-1] - 1, 1)

                        omega = omega_d
                        W_total[i,t] = (1-τ) * ω_grid[Int(omega)] * h_grid[H_total[i,t]]
                        C_total[i,t] = C[Int(i_b), T_total[i,t], H_total[i,t], ifelse(U_total[i,t] == 1, 1, 1+Int(omega))]
                        S = (W_total[i,t] + B_total[i,t] - C_total[i,t]) * (1+r)
                    elseif p_case == 2
                        U_total[i,t] = 1
                        H_total[i,t] = max(H_total[i,t-1] - 1, 1)
                        W_total[i,t] = z
                        C_total[i,t] = C[Int(i_b), T_total[i,t], H_total[i,t], ifelse(U_total[i,t] == 1, 1, 1+Int(omega))]
                        S = (W_total[i,t] + B_total[i,t] - C_total[i,t]) * (1+r)
                    elseif p_case == 3
                        U_total[i,t] = 0
                        H_total[i,t] = H_total[i,t-1]

                        omega = omega_s
                        W_total[i,t] = (1-τ) * ω_grid[Int(omega)] * h_grid[H_total[i,t]]
                        C_total[i,t] = C[Int(i_b), T_total[i,t], H_total[i,t], ifelse(U_total[i,t] == 1, 1, 1+Int(omega))]
                        S = (W_total[i,t] + B_total[i,t] - C_total[i,t]) * (1+r)
                    elseif p_case == 4
                        U_total[i,t] = 1
                        H_total[i,t] = H_total[i,t-1]
                        W_total[i,t] = z
                        C_total[i,t] = C[Int(i_b), T_total[i,t], H_total[i,t], ifelse(U_total[i,t] == 1, 1, 1+Int(omega))]
                        S = (W_total[i,t] + B_total[i,t] - C_total[i,t]) * (1+r)
                    end
                    
                elseif U_total[i, t-1] == 0
                    p_grid = [δ * p_H, δ * (1-p_H), (1-δ) * p_H, (1-δ) * (1-p_H)]
                    p_grid = cumsum(p_grid)
                    p_case = ceil(get_index(prob, p_grid))
                    
                    U_total[i,t] = ifelse(p_case ∈ [1,2], 1, 0)
                    H_total[i,t] = ifelse(p_case ∈ [1,3], min(H_total[i,t-1]+1, N_h), H_total[i,t-1])
                    W_total[i,t] = ifelse(U_total[i,t] == 1, z, (1-τ) * ω_grid[max(1,Int(omega))] * h_grid[H_total[i,t]])
                    C_total[i,t] = C[Int(i_b), T_total[i,t], H_total[i,t], ifelse(U_total[i,t] == 1, 1, 1+Int(omega))]
                    S = (W_total[i,t] + B_total[i,t] - C_total[i,t]) * (1+r)
                end

            end
        end
    end

    return B_total[:, (total-burnin+1):total], T_total[:, (total-burnin+1):total], U_total[:, (total-burnin+1):total], H_total[:, (total-burnin+1):total], W_total[:, (total-burnin+1):total], C_total[:, (total-burnin+1):total], id_total[:, (total-burnin+1):total]
end

function Run_simul(pars, res, sim)
    sim.B_sim, sim.T_sim, sim.U_sim, sim.H_sim, sim.W_sim, sim.C_sim, sim.id_sim = Simulate(pars, res)
end

## Run all

pars, res = Initialize()
VFI(pars, res)

sim = Init_sim(pars)
Run_simul(pars, res, sim)

## Combine into panel data

quarters = repeat(1:120,1000)

panel = zeros(120*1000, 8)

panel[:,1] = vec(sim.id_sim')
panel[:,2] = quarters 
panel[:,3] = vec(sim.B_sim') # Wealth
panel[:,4] = vec(sim.U_sim') # Unemployment status
panel[:,5] = vec(sim.T_sim') # Ages (in quarters)
panel[:,6] = vec(sim.H_sim') # Human capital
panel[:,7] = vec(sim.W_sim') # After tax/transfer earnings
panel[:,8] = vec(sim.C_sim') # Consumption

panel = DataFrame(panel, :auto)
rename!(panel, Symbol.(["ID", "Year", "Wealth", "Unemployment", "Age", "Human Capital", "Earnings", "Consumption"]))
CSV.write("/Users/Yeonggyu/Desktop/윤영규/대학원 (UW-Madison)/Coursework/Spring 2023/Econ 810 - Advanced Macroeconomics/Week 3/HW/Simulated panel.csv", panel, writeheader = true)
