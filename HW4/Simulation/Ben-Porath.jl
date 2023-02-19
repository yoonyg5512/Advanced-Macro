######################################################################
## Ben-Porath Model with Bewley type income process ##################
######################################################################
# Prepared by Yeonggyu Yun, Stefano Lord, and Fernando de Lima Lopes #
######################################################################

using Parameters, Statistics, Plots, CSV, Tables, Random, Distributions, DataFrames, Interpolations, Optim

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

    N_h::Int64 = 100
    N_k::Int64 = 20
    N_s::Int64 = 20
    h_grid::Array{Float64, 1} = range(start = 0.0, stop = 10.0, length = N_h)
    k_grid::Array{Float64, 1} = range(start = 0.0, stop = 20.0, length = N_k)
    s_grid::Array{Float64, 1} = range(start = 0.0, stop = 1.0, length = N_s)

    # Simulation

    N_i::Int64 = 1000 # Number of individuals to simulate
end

mutable struct Results
    # Value functions

    V::Array{Float64, 3}
    
    # Policy functions

    S::Array{Float64, 3} # Investment in human capital
    K::Array{Float64, 3} # Asset
end

function Initialize()
    pars = Params()

    V::Array{Float64, 3} = zeros(pars.N_k, pars.T, pars.N_h)
    S::Array{Float64, 3} = zeros(pars.N_k, pars.T, pars.N_h)
    K::Array{Float64, 3} = zeros(pars.N_k, pars.T, pars.N_h)

    res = Results(V, S, K)
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
    z1 = -zN

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

                        v_today(i_kp, i_sp) = (((1+r)*k + R^(t-1) * h * (1-s_interp(i_sp)) - k_interp(i_kp))^(1-σ) - 1) / (1-σ) + β * v_tomorrow(i_kp, i_sp)

                        obj(x) = - v_today(x[1], x[2])
                        lower = [1.0, 1.0]
                        upper = [Float64(get_index((1+r)*k + R^(t-1) * h, k_grid)), Float64(N_k)]
                        
                        if lower[1] == upper[2]
                            obj_t(x) = - v_today(lower[1], x)
                            opt = optimize(obj_t, lower[2], upper[2])

                            k_tomorrow = k_interp(lower[1])
                            s_today = s_interp(opt.minimizer[1])
                            v_now = -opt.minimum

                            k_cand[i_k, t, i_h] = k_tomorrow
                            s_cand[i_k, t, i_h] = s_today
                            v_cand[i_k, t, i_h] = v_now
                        else
                            opt = optimize(obj, lower, upper, [(lower[1] +upper[1]) / 2, (lower[2] + upper[2])/ 2])

                            k_tomorrow = k_interp(opt.minimizer[1])
                            s_today = s_interp(opt.minimizer[2])
                            v_now = -opt.minimum
                            
                            k_cand[i_k, t, i_h] = k_tomorrow
                            s_cand[i_k, t, i_h] = s_today
                            v_cand[i_k, t, i_h] = v_now
                        end
                    end
                end
            end
        println(t)
    end
    return k_cand, s_cand, v_cand
end

function Solve_Model(pars, res)
    res.K, res.S, res.V = Bellman(pars, res)
end

## 3. Simulation

mutable struct Sims
    E::Array{Float64, 2} # Earnings of each individual of ages
end

function Init_sim(pars)
    E::Array{Float64, 2} = zeros(pars.N_i, 30)
    A::Array{Float64, 2} = zeros(pars.N_i, 30)
end

function Simulate(pars, res)
    @unpack N_i = pars
    @unpack K, S, V = res
end
