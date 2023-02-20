######################################################################
## Ben-Porath Model with Bewley type income process ##################
######################################################################
# Prepared by Yeonggyu Yun, Stefano Lord, and Fernando de Lima Lopes #
######################################################################

using Parameters, Statistics, Plots, CSV, Tables, Random, Distributions, DataFrames, Interpolations, Optim, JuMP, Ipopt

##### 1. Housekeeping

Random.seed!(1234)

@with_kw struct Params
    T::Int64 = 30 # 30 years
    R::Float64 = 1.0019 # Rental rate to human capital 
    α::Float64 = 0.7
    r::Float64 = 0.04
    β::Float64 = 0.99 # Time discount
    σ::Float64 = 2.0 # Risk aversion

    μ_z::Float64 = -0.29
    σ_z::Float64 = sqrt(0.11)
    μ_h::Float64 = 2
    σ_h::Float64 = sqrt(0.5)

    # Grids for human capital h and saving choice b and piece-rate ω

    N_h::Int64 = 30
    N_k::Int64 = 20
    N_s::Int64 = 40
    h_grid::Array{Float64, 1} = range(start = 0.0, stop = 10.0, length = N_h)
    k_grid::Array{Float64, 1} = range(start = 0.0, stop = 20.0, length = N_k)
    s_grid::Array{Float64, 1} = range(start = 0.0, stop = 1.0, length = N_s)

    # Simulation

    N_i::Int64 = 10000 # Number of individuals to simulate
    T_sim::Int64 = 30 # Number of years to simulate
    burnin::Int64 = 45
end

mutable struct Results
    # Value functions

    V::Array{Float64, 3}
    
    # Policy functions

    S::Array{Float64, 3} # Investment in human capital
    K::Array{Float64, 3} # Asset

    # Discretized grid of z

    zs::Array{Float64, 1}
    Π_z::Array{Float64, 1}
end

function Initialize()
    pars = Params()

    V::Array{Float64, 3} = zeros(pars.N_k, pars.T, pars.N_h)
    S::Array{Float64, 3} = zeros(pars.N_k, pars.T, pars.N_h)
    K::Array{Float64, 3} = zeros(pars.N_k, pars.T, pars.N_h)
    zs::Array{Float64, 1} = zeros(20)
    Π_z::Array{Float64, 1} = zeros(20)

    res = Results(V, S, K, zs, Π_z)
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

## 2. Bellman equation

function Bellman(pars, res)
    @unpack β, σ, T, R, α, r, μ_z, σ_z, N_h, N_k, h_grid, k_grid, s_grid = pars

    ## Discretize the space of z following Tauchen
    
    N_z = 20
    zN = 3*σ_z
    z1 = -3*σ_z

    Π_z = zeros(N_z)
    zs = range(start = z1, stop = zN, length = N_z)
    d = zs[2] - zs[1]
    for (j, z_j) in enumerate(zs)
        if j == 1
            Π_z[j] = cdf(Normal(0, 1), (z_j+d/2)/σ_z)
        elseif j == N_z
            Π_z[j] = 1 - cdf(Normal(), (z_j-d/2)/σ_z)
        else
            Π_z[j] = cdf(Normal(0, 1), (z_j+d/2)/σ_z) - cdf(Normal(0, 1), (z_j-d/2)/σ_z)
        end
    end
    zs = zs .+ μ_z

    v_cand = zeros(N_k, T, N_h);
    s_cand = zeros(N_k, T, N_h);
    k_cand = zeros(N_k, T, N_h);

    k_interp = interpolate(k_grid, BSpline(Linear()))
    s_interp = interpolate(s_grid, BSpline(Linear()))

    v_last = zeros(N_k, N_h) # Value at last period (Save nothing and eat up in the last period)
    for i in 1:N_k
        for j in 1:N_h
            c_last = R^(T-1) * h_grid[j] + (1+r) * k_grid[i]
            v_last[i,j] = (c_last^(1-σ) - 1) / (1-σ)
        end
    end
    v_cand[:,T,:] = v_last;
    k_cand[:,T,:] .= 0.0; 
    s_cand[:,T,:] .= 0.0;
    v_cand[1,T,1] = 1e-18;
    
    for t in collect((T-1):-1:1)
        v_interp = interpolate(v_cand[:,t+1,:], BSpline(Linear()));

        for (i_k, k) in enumerate(k_grid)
            for (i_h, h) in enumerate(h_grid)

                function v_tomorrow(i_kp, i_sp)
                    v_t = 0.0;
                    H = h + (h * s_interp(i_sp))^α
                    
                    for (n_z, z)  in enumerate(zs) 
                        h_next = exp(z) * H
                        i_h_next = get_index(h_next, h_grid)
        
                        v_t += Π_z[n_z] * v_interp(i_kp, i_h_next)
                    end
        
                    return v_t
                end
            
                function v_today(i_sp)
                    budget = (1+r)*k + R^(t-1) * h * (1-s_interp(i_sp))
                    v_given_s(i_kp) = ((budget - k_interp(i_kp))^(1-σ) - 1) / (1 - σ) + β * v_tomorrow(i_kp, i_sp)
                    lower_k = 1.0 
                    upper_k = get_index(budget, k_grid)
                    obj_k(i_kp) = - v_given_s(i_kp)
                    opt_k = optimize(obj_k, lower_k, upper_k)
        
                    return opt_k.minimizer[1], -opt_k.minimum
                end

                obj_s(i_sp) = - v_today(i_sp)[2]
                lower_s = 1.0
                upper_s = Float64(N_k)

                opt_s = optimize(obj_s, lower_s, upper_s)
                s_today = s_interp(opt_s.minimizer[1])
                k_tomorrow = k_interp(v_today(opt_s.minimizer[1])[1])
                v_now  = - opt_s.minimum

                k_cand[i_k, t, i_h] = k_tomorrow
                s_cand[i_k, t, i_h] = s_today
                v_cand[i_k, t, i_h] = ifelse(isinf(v_now), 1e-18, v_now)
            end
        end
    end

    return k_cand, s_cand, v_cand, zs, Π_z
end

function Solve_Model(pars, res)
    res.K, res.S, res.V, res.zs, res.Π_z = Bellman(pars, res)
end

## 3. Simulation

mutable struct Sims
    E::Array{Float64, 2} # Earnings of each individual at each year
    A::Array{Int64, 2} # Ages of each individual at each year
    H::Array{Float64, 2}
    S::Array{Float64, 2}
    id::Array{Int64, 2} # Individual id
end

function Init_sim(pars)
    E::Array{Float64, 2} = zeros(pars.N_i, pars.T_sim) # 30 years of panel
    A::Array{Int64, 2} = zeros(pars.N_i, pars.T_sim)
    id::Array{Int64, 2} = zeros(pars.N_i, pars.T_sim)
    H::Array{Float64, 2} = zeros(pars.N_i, pars.T_sim)
    S::Array{Float64, 2} = zeros(pars.N_i, pars.T_sim)

    sims = Sims(E, A, H, S, id)
    return sims
end

function Simulate(pars, res)
    @unpack N_i, T_sim, burnin, μ_h, σ_h, R, N_k, T, α, h_grid, k_grid, s_grid = pars
    @unpack K, S, V, zs, Π_z = res

    burnin = 150

    E_sim = zeros(N_i, T_sim+burnin)
    A_sim = zeros(N_i, T_sim+burnin)
    id_sim = zeros(N_i, T_sim+burnin)
    H_sim = zeros(N_i, T_sim+burnin)
    S_sim = zeros(N_i, T_sim+burnin)

    KP_sim = zeros(N_i, T_sim+burnin)

    h_interp = interpolate(h_grid, BSpline(Linear()))
    K_interp = interpolate(K, BSpline(Linear()))
    S_interp = interpolate(S, BSpline(Linear()))
    s_interp = interpolate(s_grid, BSpline(Linear()))

    A_start = rand(1:T, N_i)
    A_sim[:,1] = A_start
    
    h_start = rand(Normal(μ_h, σ_h), N_i)
    h_start[h_start .< 0] .= 0.0
    H_sim[:,1] = h_start
    k_start = rand(1:N_k, N_i)

    for i in 1:N_i
        h_start[i] = get_index(h_start[i], h_grid)

        if A_start[i] == 1
            k_start[i] = 1
        end
        KP_sim[i,1] = k_start[i]

        E_sim[i,1] = R^(A_sim[i,1] - 1) * h_interp(h_start[i]) * (1 - S_interp(KP_sim[i,1], A_sim[i,1], h_start[i]))
        KP_sim[i,2] = get_index(K_interp(KP_sim[i,1], A_sim[i,1], h_start[i]), k_grid)
        S_sim[i,1] = S_interp(KP_sim[i,1], A_sim[i,1], h_start[i])
    end

    id_sim[1,1] = 1
    for i in 1:N_i
        if i > 1
            id_sim[i,1] = id_sim[i-1, (T_sim+burnin)] + 1
        end
        for j in 2:(T_sim + burnin)
            if A_sim[i,j-1] == T
                
                id_sim[i,j] = id_sim[i,j-1] + 1
                A_sim[i,j] = 1

                h_init = maximum([rand(Normal(μ_h, σ_h)), 0.0])
                h_now = get_index(h_init, h_grid)
                E_sim[i,j] = h_interp(h_now) * (1 - S_interp(1, 1, h_now))

                if j < (T_sim + burnin)
                    KP_sim[i, j+1] = get_index(K_interp(1, 1, h_now), k_grid)
                end
                
                H_sim[i,j] = h_interp(h_now)
                S_sim[i,j] = S_interp(1, 1, h_now)
            elseif A_sim[i,j-1] < T
                id_sim[i,j] = id_sim[i,j-1]

                draw_z = rand(Uniform(0,1))
                z_now = zs[Int(ceil(get_index(draw_z, cumsum(Π_z))))]
                h_now = get_index(exp(z_now) * (H_sim[i,j-1] + (H_sim[i,j-1] * S_sim[i,j-1])^α), h_grid)

                A_sim[i,j] = A_sim[i,j-1] + 1
                E_sim[i,j] = R^(A_sim[i,j] - 1) * h_interp(h_now) * (1 - S_interp(KP_sim[i,j], A_sim[i,j], h_now))
                
                if j < (T_sim + burnin)
                    KP_sim[i,j+1] = get_index(K_interp(KP_sim[i,j], A_sim[i,j], h_now), k_grid)
                end
                sp[i] = get_index(S_interp(kp[i], A_sim[i,j], h_now[i]), s_grid)

                H_sim[i,j] = h_interp(h_now)
                S_sim[i,j] = S_interp(KP_sim[i,j], A_sim[i,j], h_now)
            end
        end
    end

    return E_sim[:, (burnin+1):(T_sim+burnin)], A_sim[:, (burnin+1):(T_sim+burnin)], id_sim[:, (burnin+1):(T_sim+burnin)], H_sim[:, (burnin+1):(T_sim+burnin)], S_sim[:, (burnin+1):(T_sim+burnin)]
end

function Simulate_data(pars, res, sims)
    sims.E, sims.A, sims.id, sims.H, sims.S = Simulate(pars, res)
end

## 4. Write into panel data

pars, res = Initialize()
Solve_Model(pars, res)

sims = Init_sim(pars)
Simulate_data(pars, res, sims)

years = repeat(1:pars.T_sim, pars.N_i)
panel = zeros(pars.T_sim*pars.N_i, 6)

panel[:,1] = vec(sims.id') # ID
panel[:,2] = years # Year
panel[:,3] = vec(sims.A') # Age
panel[:,4] = vec(sims.E') # Earnings
panel[:,5] = vec(sims.S') # Earnings
panel[:,6] = vec(sims.H') # Earnings


panel = DataFrame(panel, :auto)
rename!(panel, Symbol.(["ID", "Year", "Age", "Earnings", "S", "H"]))
CSV.write("/Users/Yeonggyu/Desktop/윤영규/대학원 (UW-Madison)/Coursework/Spring 2023/Econ 810 - Advanced Macroeconomics/Week 4/HW/Simulated panel.csv", panel, writeheader = true)

mean_by_age = zeros(pars.T)

for i in 1:pars.T
    mean_by_age[i] = mean(sims.S[sims.A .== i])
end

plot(mean_by_age)