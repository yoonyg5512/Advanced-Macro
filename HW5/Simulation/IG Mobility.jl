######################################################################
## Intergenerational Mobility Model (Lee and Seshadri, 2019 JPE) #####
######################################################################
# by Yeonggyu Yun, Stefano Lord-Medrano, and Fernando de Lima Lopes ##
######################################################################

using Parameters, Statistics, Plots, CSV, Tables, Random, Distributions, DataFrames, Interpolations, Optim

##### 1. Housekeeping

Random.seed!(1234)

@with_kw struct Params
    T::Int64 = 9 # 9 periods in life
    r::Float64 = (1.04)^6 - 1 # Interst rate
    β::Float64 = (1 / 1.04)^6 # Time discount
    σ::Float64 = 2.0 # Risk aversion
    θ::Float64 = 0.32 # Altruism

    κ::Array{Float64, 1} = [9.2298, 9.6825, 10.106, 10.476, 10.743, 10.954, 11.108, 11.23, 11.426] .- 9.2298 # Age specific income for periods 4-12
    ω_c::Float64 = 0.5
    ρ_h::Float64 = 0.97^6 # Can choose different values!
    σ_h::Float64 = 0.3
    N_η::Int64 = 20

    # Grids for parents human capital h and kids human capital hc

    N_h::Int64 = N_η
    N_hc::Int64 = N_η
    h_grid::Array{Float64, 1} = range(start = 0.0, stop = 1.0, length = N_h)
    hc_grid::Array{Float64, 1} = range(start = 0.0, stop = 1.0, length = N_hc)

    # Grids for asset b, investment in kids i, transfer to kids τ

    N_b::Int64 = 11
    N_τ::Int64 = 10
    N_i::Int64 = 10
    b_grid::Array{Float64, 1} = range(start = -exp(5), stop = exp(5), length = N_b)
    τ_grid::Array{Float64, 1} = range(start = 0.0, stop =  exp(3), length = N_τ)
    i_grid::Array{Float64, 1} = range(start = 0.0, stop = exp(3), length = N_i)

    # Simulation

    N_ind::Int64 = 10000 # Number of individuals to simulate
    T_sim::Int64 = 9 # Number of periods to simulate
end

mutable struct Results
    # Value functions

    V::Array{Float64, 3} # Value function for periods 4, 10, 11, 12
    Vc::Array{Float64, 4} # Value function for periods 5, 6, 7, 8, 9
    
    # Policy functions

    B::Array{Float64, 3} # Asset choice in periods 4, 10, 11, 12
    Bc::Array{Float64, 4} # Asset choice in peridos 5, 6, 7, 8, 9
    I::Array{Float64, 4} # Investment choice in periods 5, 6, 7, 8
    Tr::Array{Float64, 3} # Transfer to kids in period 9

    # Discretized grid of z

    ηs::Array{Float64, 1}
    Π_η::Array{Float64, 2}
end

function Initialize()
    pars = Params()

    V::Array{Float64, 3} = zeros(pars.N_b, 4, pars.N_h) # Value function for periods 4, 10, 11, 12
    Vc::Array{Float64, 4} = zeros(pars.N_b, 5, pars.N_h, pars.N_hc) # Value function for periods 5, 6, 7, 8, 9
    
    # Policy functions

    B::Array{Float64, 3} = zeros(pars.N_b, 4, pars.N_h) # Asset choice in periods 4, 10, 11, 12
    Bc::Array{Float64, 4} = zeros(pars.N_b, 5, pars.N_h, pars.N_hc) # Asset choice in peridos 5, 6, 7, 8, 9
    I::Array{Float64, 4} = zeros(pars.N_b, 4, pars.N_h, pars.N_hc) # Investment choice in periods 5, 6, 7, 8
    Tr::Array{Float64, 3} = zeros(pars.N_b, pars.N_h, pars.N_hc) # Transfer to kids in period 9

    # Discretized grid of z

    ηs::Array{Float64, 1} = zeros(pars.N_η)
    Π_η::Array{Float64, 2} = zeros(pars.N_η, pars.N_η)

    res = Results(V, Vc, B, Bc, I, Tr, ηs, Π_η)
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
    @unpack σ_h, ρ_h, N_η = pars

    ## Discretize the space of permanent component

    zN = 3*(σ_h^2 / (1-ρ_h^2))^(1/2)
    z1 = -zN

    Π_η = zeros(N_η, N_η)

    ηs = range(start = z1, stop = zN, length = N_η)
    d1 = ηs[2] - ηs[1]
    for (j, z_j) in enumerate(ηs)
        for (k, z_k) in enumerate(ηs)
            if k == 1
                Π_η[j,k] = cdf(Normal(0, 1), (z_k+d1/2-ρ_h*z_j)/σ_h)
            elseif k == N_η
                Π_η[j,k] = 1 - cdf(Normal(0,1), (z_k-d1/2-ρ_h*z_j)/σ_h)
            else
                Π_η[j,k] = cdf(Normal(0, 1), (z_k+d1/2-ρ_h*z_j)/σ_h) - cdf(Normal(0, 1), (z_k-d1/2-ρ_h*z_j)/σ_h)
            end
        end
    end
    return ηs, Π_η
end

## 2. Bellman equation

function Bellman(pars, res)
    @unpack β, r, σ, T, θ, κ, ω_c, h_grid, hc_grid, b_grid, τ_grid, i_grid, N_hc = pars
    @unpack V, Vc = res
    ## Discretize the space of z following Tauchen
    
    res.ηs, res.Π_η = trans_Tauchen(pars)
    @unpack ηs, Π_η = res

    ## Initialize 

    V_cand = zeros(pars.N_b, 4, pars.N_h) # Value function for periods 4, 10, 11, 12
    Vc_cand = zeros(pars.N_b, 5, pars.N_h, pars.N_hc) # Value function for periods 5, 6, 7, 8, 9
    B = zeros(pars.N_b, 4, pars.N_h) # Asset choice in periods 4, 10, 11, 12
    Bc = zeros(pars.N_b, 5, pars.N_h, pars.N_hc) # Asset choice in peridos 5, 6, 7, 8, 9
    I = zeros(pars.N_b, 4, pars.N_h, pars.N_hc) # Investment choice in periods 5, 6, 7, 8
    Tr = zeros(pars.N_b, pars.N_h, pars.N_hc) # Transfer to kids in period 9

    b_interp = interpolate(b_grid, BSpline(Linear()))
    τ_interp = interpolate(τ_grid, BSpline(Linear()))
    i_interp = interpolate(i_grid, BSpline(Linear()))
    V_interp = interpolate(V, BSpline(Linear()))
    Vc_interp = interpolate(Vc, BSpline(Linear()))

    ## Periods 10, 11, 12

    for (i_b, b) in enumerate(b_grid)
        for (i_h, h) in enumerate(h_grid)
            B[i_b, 4, i_h] = 0
            c = exp(κ[9] + h) + (1+r) * b
            V_cand[i_b, 4, i_h] = ifelse(c > 0, (c^(1-σ) - 1) / (1 - σ), -1e+8)
        end
    end

    function v_tomorrow(i_bp, i_h, t)
        v = 0.0
        for (i_hp, hp) in enumerate(h_grid)
            v += Π_η[i_h, i_hp] * V_interp(i_bp, t+1, i_hp) 
        end
        return v
    end

    for t in collect(3:-1:2)
        for (i_b, b) in enumerate(b_grid)
            for (i_h, h) in enumerate(h_grid)

                budget = exp(κ[t+5] + h) + (1+r) * b
                
                v_today(i_bp) = ifelse(budget - b_interp(i_bp) > 0, ((budget - b_interp(i_bp))^(1-σ) -1) / (1-σ) + β * v_tomorrow(i_bp, i_h, t), -1e+8)
                obj(i_bp) = - v_today(i_bp) 
                lower = 1.0 
                upper = get_index(budget, b_grid)
                opt = optimize(obj, lower, upper)

                B[i_b, t, i_h] = opt.minimizer[1]
                V_cand[i_b, t, i_h] = - opt.minimum
            end
        end
    end

    ## Period 9

    function v_tomorrow(i_bp, i_h)
        v = 0
        for (i_hp, hp) in enumerate(h_grid)
            v += Π_η[i_h, i_hp] * V_interp(i_bp, 2, i_hp)
        end
        return v
    end

    function v_now(i_τ, h, b, i_h, i_hc)
        budget = exp(κ[6] + h) + (1+r) * b - τ_interp(i_τ)
        i_τ_in_b = get_index(τ_interp(i_τ), b_grid)
        v_t(i_bp) = ifelse(budget - b_interp(i_bp) > 0, ((budget - b_interp(i_bp))^(1-σ) - 1)/(1-σ) + θ * V_interp(i_τ_in_b, 1, i_hc) + β * v_tomorrow(i_bp, i_h), -1e+8)
        obj_9(i_bp) = - v_t(i_bp)
        lower = 1.0
        upper = get_index(budget, b_grid)
        opt = optimize(obj_9, lower, upper)

        return opt.minimizer[1], - opt.minimum
    end

    for (i_b, b) in enumerate(b_grid)
        for (i_h, h) in enumerate(h_grid)
            for (i_hc, hc) in enumerate(hc_grid)
                obj_τ(i_τ) = - v_now(i_τ, h, b, i_h, i_hc)[2]
                lower = 1.0
                upper = get_index(exp(κ[6] + h) + (1+r) * b, τ_grid)
                opt_τ = optimize(obj_τ, lower, upper)

                Tr[i_b, i_h, i_hc] = opt_τ.minimizer[1]
                Bc[i_b, 5, i_h, i_hc] = v_now(opt_τ.minimizer[1], h, b, i_h, i_hc)[1]
                Vc_cand[i_b, 5, i_h, i_hc] = - opt_τ.minimum
            end
        end
    end

    ## Periods 5, 6, 7, 8
    function v_tomorrow(i_bp, i_i, hc, i_h, t)
        v = 0
        i_hcp = get_index((1-ω_c) * hc + ω_c * log(i_interp(i_i) + 1.0), hc_grid)

        for (i_hp, hp) in enumerate(h_grid)
            v += Π_η[i_h, i_hp] * Vc_interp(i_bp, t+1, i_hp, i_hcp)
        end
        return v
    end

    function v_now(i_i, h, b, hc, i_h, t)
        budget = exp(κ[t+1] + h) + (1+r) * b - i_interp(i_i)
        v_t(i_bp) = ifelse(budget - b_interp(i_bp) > 0, ((budget - b_interp(i_bp))^(1-σ) - 1)/(1-σ) + β * v_tomorrow(i_bp, i_i, hc, i_h, t), -1e+8)
        obj_c(i_bp) = - v_t(i_bp)
        lower = 1.0
        upper = get_index(budget, b_grid)
        opt = optimize(obj_c, lower, upper)

        return opt.minimizer[1], - opt.minimum
    end

    for t in collect(4:-1:1)
        for (i_b, b) in enumerate(b_grid)
            for (i_h, h) in enumerate(h_grid)
                for (i_hc, hc) in enumerate(hc_grid)
                    obj_i(i_i) = - v_now(i_i, h, b, hc, i_h, t)[2]
                    lower = 1.0
                    upper = get_index(exp(κ[t+1] + h) + (1+r) * b, i_grid)
                    opt_i = optimize(obj_i, lower, upper)
    
                    I[i_b, t, i_h, i_hc] = opt_i.minimizer[1]
                    Bc[i_b, t, i_h, i_hc] = v_now(opt_i.minimizer[1], h, b, hc, i_h, t)[1]
                    Vc_cand[i_b, t, i_h, i_hc] = - opt_i.minimum
                end
            end
        end
    end

    ## Period 4
    function v_tomorrow(i_bp, i_h)
        v = 0.0
        for (i_hp, hp) in enumerate(h_grid)
            for i_hpc in 1:Int(N_hc / 2)
                v += Π_η[i_h, i_hp] * 2 / N_hc * Vc_interp(i_bp, 1, i_hp, i_hpc)
            end 
        end
        return v
    end
    
    for (i_b, b) in enumerate(b_grid)
        for (i_h, h) in enumerate(h_grid)
            budget = exp(κ[1] + h) + (1+r) * b
          
            v_today(i_bp) = ifelse(budget - b_interp(i_bp) > 0, ((budget - b_interp(i_bp))^(1-σ) -1) / (1-σ) + β * v_tomorrow(i_bp, i_h), -1e+8)
            obj(i_bp) = - v_today(i_bp) 
            lower = 1.0 
            upper = get_index(budget, b_grid)
            opt = optimize(obj, lower, upper)

            B[i_b, 1, i_h] = opt.minimizer[1]
            V_cand[i_b, 1, i_h] = - opt.minimum
        end
    end

    return V_cand, Vc_cand, B, Bc, I, Tr
end

function Solve_Model(pars, res)
    tol = 1e-5
    err = 100.0
    n = 1

    while err > tol
        V_cand, Vc_cand, B_cand, Bc_cand, I_cand, Tr_cand = Bellman(pars, res)
        err = findmax([maximum(abs.(res.V .- V_cand)), maximum(abs.(res.Vc .- Vc_cand))])[1]
        err_ind = findmax([maximum(abs.(res.V .- V_cand)), maximum(abs.(res.Vc .- Vc_cand))])[2]
        res.V = V_cand
        res.Vc = Vc_cand
        res.B = B_cand
        res.Bc = Bc_cand
        res.I = I_cand
        res.Tr = Tr_cand
        println("Iteration ", n, ", Error: ", err, " on matrix ", err_ind)
        
        n += 1
    end
end

## 3. Simulation

mutable struct Sims
    E::Array{Float64, 2} # Earnings of each individual in each period
    H::Array{Float64, 2} # Human capital of each individual in each period
    I::Array{Float64, 2}
    Tr::Array{Float64}
    Hc::Array{Float64, 2}
    B::Array{Float64, 2}
end

function Init_sims(pars)
    E::Array{Float64, 2} = zeros(pars.N_ind, T_sim)
    H::Array{Float64, 2} = zeros(pars.N_ind, T_sim)
    I::Array{Float64, 2} = zeros(pars.N_ind, 4)
    Tr::Array{Float64} = zeros(pars.N_ind)
    Hc::Array{Float64, 2} = zeros(pars.N_ind, 5)
    B::Array{Float64, 2} = zeros(pars.N_ind, T_sim)

    sims = Sims(id, E, H, I, Tr, Hc, B)
end

function Simulate(pars, res)
    @unpack N_ind, T_sim, T, h_grid, i_grid, τ_grid, b_grid, N_h, N_i, N_τ, N_b, κ, ω_c = pars
    @unpack ηs, Π_η, B, Bc, I, Tr = res

    B_interp = interpolate(B, BSpline(Linear()))
    Bc_interp = interpolate(Bc, BSpline(Linear()))
    I_interp = interpolate(I, BSpline(Linear()))
    Tr_interp = interpolate(Tr, BSpline(Linear()))
    h_interp = interpolate(h_grid, BSpline(Linear()))
    b_interp = interpolate(b_grid, BSpline(Linear()))
        
    E_sim = zeros(N_ind, T_sim)
    H_sim = zeros(N_ind, T_sim)
    I_sim = zeros(N_ind, 4)
    Tr_sim = zeros(N_ind)
    Hc_sim = zeros(N_ind, 5)
    B_sim = zeros(N_ind, T_sim)

    for i in 1:N_ind
        
        # Period 4

        h_init = rand(1:N_h)
        H_sim[i,1] = h_interp(h_init)
        E_sim[i,1] = exp(H_sim[i,1] + κ[1])
        B_sim[i,1] = b_interp(rand(1:N_b))

        # Periods 5, 6, 7, 8

        for j in 2:5 
            prob = rand(Uniform(0,1))
            probs = cumsum(Π_η[Int(get_index(H_sim[i,j-1], h_grid)),:])
            case = ceil(get_index(prob, probs))

            H_sim[i,j] = h_interp(Int(case))
            E_sim[i,j] = exp(H_sim[i,j] + κ[j])
            if j == 2
                B_sim[i,j] = B_interp(get_index(B_sim[i,j-1], b_grid), 1, get_index(H_sim[i,j-1], h_grid))
                hc_init = rand(1:(N_h/2))
                I_sim[i,j-1] = i_interp(get_index(B_sim[i,j], b_grid), j-1, case, hc_init)
                Hc_sim[i,j-1] = h_interp(hc_init)
            else
                B_sim[i,j] = Bc_interp(get_index(B_sim[i,j-1], b_grid), j-2, get_index(H_sim[i,j-1], h_grid), get_index(Hc_sim[i,j-2], h_grid))
                Hc_sim[i,j-1] = (1-ω_c) * Hc_sim[i,j-2] + ω_c * log(1+I_sim[i,j-2])
                I_sim[i,j-1] = i_interp(get_index(B_sim[i,j], b_grid), j-1, case, get_index(Hc_sim[i,j-1], h_grid))
            end
        end

        # Period 9

        prob = rand(Uniform(0,1))
        probs = cumsum(Π_η[Int(get_index(H_sim[i,5], h_grid)),:])
        case = ceil(get_index(prob, probs))
        
        H_sim[i,6] = h_interp(Int(case))
        E_sim[i,6] = exp(H_sim[i,6] + κ[6])
        B_sim[i,6] = Bc_interp(get_index(B_sim[i,5], b_grid), 4, get_index(H_sim[i,5], h_grid), get_index(Hc_sim[i,4], h_grid))
        Hc_sim[i,5] = (1-ω_c) * Hc_sim[i,4] + ω_c * log(1+I_sim[i,4])
        Tr_sim[i] = Tr_interp(get_index(B_sim[i,6], b_grid), get_index(H_sim[i,6], h_grid), get_index(Hc_sim[i,5], h_grid))        

        # Period 10, 11, 12
        
        for j in 7:9
            prob = rand(Uniform(0,1))
            probs = cumsum(Π_η[Int(get_index(H_sim[i,j-1], h_grid)),:])
            case = ceil(get_index(prob, probs))
            
            H_sim[i,j] = h_interp(Int(case))
            E_sim[i,j] = exp(H_sim[i,j] + κ[j])
            if j == 7
                B_sim[i,j] = Bc_interp(get_index(B_sim[i,j-1], b_grid), 5, get_index(H_sim[i,j-1], h_grid), get_index(Hc_sim[i,j-2], h_grid))
            else
                B_sim[i,j] = B_interp(get_index(B_sim[i,j-1], b_grid), j-6, get_index(H_sim[i,j-1], h_grid))
            end

        end
    end

    return E_sim, H_sim, I_sim, Tr_sim, Hc_sim, B_sim
end

function Simulate_data(pars, res, sims)
    sims.E, sims.H, sims.I, sims.Tr, sims.Hc, sims.B = Simulate(pars, res)
end

## 4. Write into panel data

pars, res = Initialize()
Solve_Model(pars, res)

sims = Init_sim(pars)
Simulate_data(pars, res, sims)

years = repeat(1:pars.T_sim, pars.N_ind)
panel = zeros(pars.T_sim*pars.N_ind, 5)

panel[:,1] = vec(sims.id') # ID
panel[:,2] = years # Year
panel[:,3] = vec(sims.A') # Age
panel[:,4] = vec(sims.E') # Earnings
panel[:,5] = vec(sims.H') # Human capital


panel = DataFrame(panel, :auto)
rename!(panel, Symbol.(["ID", "Year", "Age", "Earnings", "Human capital"]))
CSV.write("/Users/Yeonggyu/Desktop/윤영규/대학원 (UW-Madison)/Coursework/Spring 2023/Econ 810 - Advanced Macroeconomics/Week 4/HW/Simulated panel.csv", panel, writeheader = true)
