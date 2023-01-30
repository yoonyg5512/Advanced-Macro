######################################################################
##### Replication file for Kaplan and Violante (2010, AEJ:Macro) #####
######################################################################
# Prepared by Yeonggyu Yun, Stefano Lord, and Fernando de Lima Lopes #
######################################################################

using Parameters, Statistics, Random, Distributions, Interpolations, Optim, Plots, JLD2

## We used zero borrowing limit for simplicity.

##### 1. Housekeeping

Random.seed!(1234)

@with_kw struct Params
    β::Float64 = 0.975 # Time-discount factor
    γ::Float64 = 2.0 # CRRA
    r::Float64 = 0.04 # Real interest rate
    T::Int64 = 35 # Age

    κ::Array{Float64, 1} = [9.6980114, 9.7620722, 9.8271754, 9.8870045, 9.9505494, 10.0128446, 10.0730949, 10.1379952, 10.1938201, 10.2577349, 10.315236, 10.3790348, 10.4361798, 10.4731184, 10.5280191, 10.5683963, 10.6206219, 10.650484, 10.687635, 10.719778, 10.761518, 10.789204, 10.833912, 10.860458, 10.904908, 10.942486, 10.979517, 11.014085, 11.040553, 11.069921, 11.119904, 11.141226, 11.194406, 11.260357, 11.295121] # Age-dependent income
    σ_0::Float64 = sqrt(0.15) # Volatility of initial permanent income
    σ_η::Float64 = sqrt(0.0166) # Volatility of permanent income shock
    σ_ϵ::Float64 = sqrt(0.0174) # Volatility of transitory income shock
    ρ::Float64 = 0.97 # Persistence of permanent income

    m::Float64 = 3.0 # Tauchen maximum Value
    N_A::Int64 = 30 # Grid size for asset 
    N_perp::Int64 = 39 # Grid size for permanent income
    N_trans::Int64 = 19 # Grid size for transitory income

    N_pop::Int64 = 1000 # Number of individuals to simulate
end

mutable struct Results
    As::Array{Float64, 1} # Asset grid depending on the borrowing limit

    V::Array{Float64, 4} # Value function
    A::Array{Float64, 4} # Savings policy function
    Π_perp::Array{Float64, 2} # Transition probability matrix
    Π_trans::Array{Float64, 2} # Transition probability matrix
    zs::Array{Float64, 1} # States of permanent components
    es::Array{Float64, 1} # States of transitory components

    path_sim::Array{Float64, 3} # Simulated data with income paths (indices to use)
    value_sim::Array{Float64, 2} # Simulated data of values
    con_sim::Array{Float64, 2} # Simulated data of consumption
    sav_sim::Array{Float64, 2} # Simulated data of savings
    inc_sim::Array{Float64, 2} # Simulated data of earnings
    per_sim::Array{Float64, 2} # Simulated persistent income (z's)
    tra_sim::Array{Float64, 2} # Simulated transitory income (ϵ's)
end

function Initialize()
    pars = Params()

    As::Array{Float64, 1} = range(start = 0.0, stop = 300000.0, length = pars.N_A) # maximum level of asset following KV

    V::Array{Float64, 4} = zeros(pars.N_A, pars.T, pars.N_perp, pars.N_trans)
    A::Array{Float64, 4} = zeros(pars.N_A, pars.T, pars.N_perp, pars.N_trans)
    Π_perp::Array{Float64, 2} = zeros(pars.N_perp, pars.N_perp)
    Π_trans::Array{Float64, 2} = zeros(pars.N_trans, pars.N_trans)
    zs::Array{Float64, 1} = zeros(pars.N_perp)
    es::Array{Float64, 1} = zeros(pars.N_trans)

    path_sim::Array{Float64, 3} = zeros(pars.N_pop, pars.T, 2)
    value_sim::Array{Float64, 2} = zeros(pars.N_pop, pars.T)
    con_sim::Array{Float64, 2} = zeros(pars.N_pop, pars.T)
    sav_sim::Array{Float64, 2} = zeros(pars.N_pop, pars.T)
    inc_sim::Array{Float64, 2} = zeros(pars.N_pop, pars.T)
    per_sim::Array{Float64, 2} = zeros(pars.N_pop, pars.T)
    tra_sim::Array{Float64, 2} = zeros(pars.N_pop, pars.T)

    res = Results(As, V, A, Π_perp, Π_trans, zs, es, path_sim, value_sim, con_sim, sav_sim, inc_sim, per_sim, tra_sim)
    pars, res
end

##### 1-1. Miscellaneous functions needed

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

##### 2. Discretization (Tauchen)

## It seems like Kaplan and Violante (2010) truncated the distribution to values between 5th and 95th quantile, but not here.

function trans_Tauchen(pars)
    @unpack σ_η, σ_ϵ, ρ, m, N_perp, N_trans = pars

    ## Discretize the space of permanent component

    zN = m*(σ_η^2 / (1-ρ^2))^(1/2)
    z1 = -zN

    Π_perp = zeros(N_perp, N_perp)

    zs = range(start = z1, stop = zN, length = N_perp)
    d1 = zs[2] - zs[1]
    for (j, z_j) in enumerate(zs)
        for (k, z_k) in enumerate(zs)
            if k == 1
                Π_perp[j,k] = cdf(Normal(0, 1), (z_k+d1/2-ρ*z_j)/σ_η)
            elseif k == N_perp
                Π_perp[j,k] = 1 - cdf(Normal(), (z_k-d1/2-ρ*z_j)/σ_η)
            end
            Π_perp[j,k] = cdf(Normal(0, 1), (z_k+d1/2-ρ*z_j)/σ_η) - cdf(Normal(0, 1), (z_k-d1/2-ρ*z_j)/σ_η)
        end
    end

    ## Discretize the space of transitory component

    eN = m*σ_ϵ
    e1 = -eN

    Π_trans = zeros(N_trans, N_trans)

    es = range(start = e1, stop = eN, length = N_trans)
    d2 = es[2] - es[1]
    for (j, e_j) in enumerate(es)
        for (k, e_k) in enumerate(es)
            if k == 1
                Π_trans[j,k] = cdf(Normal(0, 1), (e_k+d2/2)/σ_ϵ)
            elseif k == N_trans
                Π_trans[j,k] = 1 - cdf(Normal(), (e_k-d2/2)/σ_ϵ)
            end
            Π_trans[j,k] = cdf(Normal(0, 1), (e_k+d2/2)/σ_ϵ) - cdf(Normal(0, 1), (e_k-d2/2)/σ_ϵ)
        end
    end

    return Π_perp, Π_trans, zs, es
end

##### 3. Solving for value and policy functions for each state

function Bellman(pars, res)
    @unpack N_A, T, N_perp, N_trans, κ, r, β, γ = pars
    @unpack Π_perp, Π_trans, zs, es, As = res

    a_cand = zeros(N_A, T, N_perp, N_trans);
    v_cand = zeros(N_A, T, N_perp, N_trans);

    a_interp = interpolate(As, BSpline(Linear()))

    v_last = zeros(N_A, N_perp, N_trans); # Value at last period (Save nothing and eat up in the last period)
    for i in 1:N_perp
        for j in 1:N_trans
            for k in 1:N_A
                v_last[k,i,j] = ((1+r)*As[k] + exp(κ[T] + zs[i] + es[j]))^(1-γ) / (1-γ) 
            end
        end
    end
    v_cand[:,T,:,:] = v_last;
    a_cand[:,T,:,:] .= 0.0; 
    
    for t in collect((T-1):-1:1)
        v_interp = interpolate(v_cand[:,t+1,:,:], BSpline(Linear()));
            for (i_z, z_today) in enumerate(zs)
                for (i_e, e_today) in enumerate(es)
                    for (i_a, a_today) in enumerate(As)
                        budget_today = (1+r) * a_today + exp(κ[t] + z_today + e_today)
                        
                        function v_tomorrow(i_ap, ip, it)
                            v_t = 0.0;
                            for (i_zt, z_tom) in enumerate(zs)
                                for (i_et, e_tom) in enumerate(es)
                                    v_t += Π_perp[ip,i_zt] * Π_trans[it,i_et] * v_interp(i_ap,i_zt,i_et)
                                end
                            end
                            return v_t
                        end
                        v_today(i_ap) = (budget_today - a_interp(i_ap))^(1-γ) / (1-γ) + β * v_tomorrow(i_ap,i_z,i_e)

                        obj(i_ap) = - v_today(i_ap)
                        lower = 1.0
                        upper = get_index(budget_today, As)

                        opt = optimize(obj, lower, upper)

                        a_tomorrow = a_interp(opt.minimizer[1])
                        v_now = -opt.minimum

                        a_cand[i_a, t, i_z, i_e] = a_tomorrow
                        v_cand[i_a, t, i_z, i_e] = v_now
                    end
                end
            end
        println(t)
    end
    return v_cand, a_cand
end

function Solve_Model(pars, res, tol::Float64 = 1e-4)
    res.Π_perp, res.Π_trans, res.zs, res.es = trans_Tauchen(pars) # Discretization
    
    res.V, res.A = Bellman(pars, res); # Backward Induction (no need to iterate)
end

##### 4. Simulation

function draw_shock(pars, res)
    @unpack N_pop, T, σ_0 = pars
    @unpack Π_perp, Π_trans, zs, es = res

    z_0 = rand(Normal(0, σ_0), N_pop) # initial persistent income z_0
    
    for i in 1:N_pop
        z_0[i] = findmax(-abs.(z_0[i] .- zs))[2] # map z_0 onto the discretized set of zs
    end

    for i in 1:N_pop
        shocks_gen = reshape(rand(Uniform(0, 1), 2*T), T, 2)

        for j in 1:T
            if j == 1
                res.path_sim[i,j,1] = Int(ceil(get_index(shocks_gen[j,1], cumsum(Π_perp[Int(z_0[i]),:]))))
                res.per_sim[i,j] = zs[Int(res.path_sim[i,j,1])]
            elseif j > 1
                res.path_sim[i,j,1] = Int(ceil(get_index(shocks_gen[j,1], cumsum(Π_perp[Int(res.path_sim[i,j-1,1]),:]))))
                res.per_sim[i,j] = zs[Int(res.path_sim[i,j,1])]
            end
            
            res.path_sim[i,j,2] = Int(ceil(get_index(shocks_gen[j,2], cumsum(Π_trans[1,:]))))
            res.tra_sim[i,j] = es[Int(res.path_sim[i,j,2])]
        end
    end
end

function Simulate(pars, res)
    draw_shock(pars, res)

    @unpack T, N_pop, N_trans, N_perp, κ, r = pars
    @unpack A, V, path_sim, per_sim, tra_sim, As = res

    v_interp = interpolate(V, BSpline(Linear()))
    a_interp = interpolate(A, BSpline(Linear()))

    for i in 1:N_pop
        for j in 1:T
            if j == 1
                res.value_sim[i,j] = V[1,j,Int(path_sim[i,j,1]), Int(path_sim[i,j,2])]
                res.inc_sim[i,j] = exp(κ[j] + per_sim[i,j] + tra_sim[i,j])
                res.sav_sim[i,j] = A[1,j,Int(path_sim[i,j,1]), Int(path_sim[i,j,2])]
                res.con_sim[i,j] = res.inc_sim[i,j] - res.sav_sim[i,j]
            elseif j > 1
                k = get_index(res.sav_sim[i,j-1], As)
                res.value_sim[i,j] = v_interp(k, j, Int(path_sim[i,j,1]), Int(path_sim[i,j,2]))
                res.inc_sim[i,j] = exp(κ[j] + per_sim[i,j] + tra_sim[i,j])
                res.sav_sim[i,j] = a_interp(k, j, Int(path_sim[i,j,1]), Int(path_sim[i,j,2]))
                res.con_sim[i,j] = res.inc_sim[i,j] + (1 + r) * res.sav_sim[i,j-1] - res.sav_sim[i,j]
            end
        end
    end
end

##### 5. Run

pars, res = Initialize()
Solve_Model(pars, res) # solve for value and policy functions

Simulate(pars, res)

