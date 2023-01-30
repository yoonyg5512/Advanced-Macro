######################################################################
##### Replication file for Kaplan and Violante (2010, AEJ:Macro) #####
######################################################################
# Prepared by Yeonggyu Yun, Stefano Lord, and Fernando de Lima Lopes #
######################################################################

using Parameters, Statistics, Random, Distributions, Interpolations, Optim

## We used zero borrowing limit for simplicity.

##### 1. Housekeeping

@with_kw struct Params
    β::Float64 = 0.975 # Time-discount factor
    γ::Float64 = 2.0 # CRRA
    r::Float64 = 0.04 # Real interest rate
    T::Int64 = 35 # Age
    κ::Array{Float64, 1} = zeros(T) # Age-dependent income
    σ_0::Float64 = 0.15 # Volatility of initial permanent income
    σ_η::Float64 = 0.01 # Volatility of permanent income shock
    σ_ϵ::Float64 = 0.05 # Volatility of transitory income shock
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

    path_sim::Array{Float64, 3} # Simulated data with income paths
    value_sim::Array{Float64, 2} # Simulated data of values
    con_sim::Array{Float64, 2} # Simulated data of consumption
    sav_sim::Array{Float64, 2} # Simulated data of savings
    inc_sim::Array{Float64, 2} # Simulated data of earnings
    per_sim::Array{Float64, 2} # Simulated persistent income
    tra_sim::Array{Float64, 2} # Simulated transitory income
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
                Π_perp[j,k] = cdf(Normal(), (z_k+d1/2-ρ*z_j)/σ_η)
            elseif k == N_perp
                Π_perp[j,k] = 1 - cdf(Normal(), (z_k-d1/2-ρ*z_j)/σ_η)
            end
            Π_perp[j,k] = cdf(Normal(), (z_k+d1/2-ρ*z_j)/σ_η) - cdf(Normal(), (z_k-d1/2-ρ*z_j)/σ_η)
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
                Π_trans[j,k] = cdf(Normal(), (e_k+d2/2)/σ_ϵ)
            elseif k == N_trans
                Π_trans[j,k] = 1 - cdf(Normal(), (e_k-d2/2)/σ_ϵ)
            end
            Π_trans[j,k] = cdf(Normal(), (e_k+d2/2)/σ_ϵ) - cdf(Normal(), (e_k-d2/2)/σ_ϵ)
        end
    end

    return Π_perp, Π_trans, zs, es
end

##### 3. Solving for value and policy functions for each state

function Bellman(pars, res)
    @unpack N_A, T, N_perp, N_trans, κ, r, β = pars
    @unpack Π_perp, Π_trans, zs, es, As = res

    a_cand = zeros(N_A, T, N_perp, N_trans);
    v_cand = zeros(N_A, T, N_perp, N_trans);

    a_interp = interpolate(As, BSpline(Linear()))

    v_last = zeros(N_A, N_perp, N_trans);
    for i in 1:N_perp
        for j in 1:N_trans
            for k in 1:N_A
                v_last[k,i,j] = - ((1+r)*As[k] + exp(κ[T] + zs[i] + es[j]))^(-1) 
            end
        end
    end
    v_interp_l = interpolate(v_last, BSpline(Linear()));

    for t in collect(T:-1:1)
        if t == T
            for (i_z, z_today) in enumerate(zs)
                for (i_e, e_today) in enumerate(es)
                    for (i_a, a_today) in enumerate(As)
                        budget_today = (1+r) * a_today + exp(κ[t] + z_today + e_today)
                        
                        function v_tomorrow(i_ap, ip, it)
                            v_t = 0.0;
                            for (i_zt, z_tom) in enumerate(zs)
                                for (i_et, e_tom) in enumerate(es)
                                    v_t += Π_perp[ip,i_zt] * Π_trans[it,i_et] * v_interp_l(i_ap,i_zt,i_et)
                                end
                            end
                            return v_t
                        end
                        v_today(i_ap) = - (budget_today - a_interp(i_ap))^(-1) + β * v_tomorrow(i_ap,i_z,i_e)

                        obj(i_ap) = - v_today(i_ap)
                        lower = 1.0
                        upper = get_index(budget_today, As)

                        opt = optimize(obj, lower, upper)

                        a_tomorrow = a_interp(opt.minimizer[1])
                        v_now = -opt.minimum

                        a_cand[i_a, t, i_z, i_e] = a_tomorrow
                        v_cand[i_a, t, i_z, i_e] = v_now
                        println([t,i_z,i_e,i_a])
                    end
                end
            end
        elseif t < T
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
                        v_today(i_ap) = - (budget_today - a_interp(i_ap))^(-1) + β * v_tomorrow(i_ap,i_z,i_e)

                        obj(i_ap) = - v_today(i_ap)
                        lower = 1.0
                        upper = get_index(budget_today, As)

                        opt = optimize(obj, lower, upper)

                        a_tomorrow = a_interp(opt.minimizer[1])
                        v_now = -opt.minimum

                        a_cand[i_a, t, i_z, i_e] = a_tomorrow
                        v_cand[i_a, t, i_z, i_e] = v_now
                        println([t,i_z,i_e,i_a])
                    end
                end
            end
        end
    end
    return v_cand, a_cand
end

function Solve_Model(pars, res, tol::Float64 = 1e-4)
    res.Π_perp, res.Π_trans, res.zs, res.es = trans_Tauchen(pars) # Discretization
    
    res.V, res.A = Bellman(pars, res) # Backward Induction (no need to iterate)
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
                res.path_sim[i,j,1] = Int(ceil(get_index(shocks_gen[j,1], cumsum(Π_perp[z_0[i],:]))))
                res.per_sim[i,j] = zs[res.path_sim[i,j,1]]
            elseif j > 1
                res.path_sim[i,j,1] = Int(ceil(get_index(shocks_gen[j,1], cumsum(Π_perp[res.path_sim[i,j-1,1],:]))))
                res.per_sim[i,j] = zs[res.path_sim[i,j,1]]
            end
            
            res.path_sim[i,j,2] = Int(ceil(get_index(shocks_gen[j,2], cumsum(Π_trans[1,:]))))
            res.tra_sim[i,j] = es[res.path_sim[i,j,2]]
        end
    end
end

function simulate(pars, res)

end

##### 5. Run

pars, res = Initialize()
Solve_Model(pars, res) # solve for value and policy functions
