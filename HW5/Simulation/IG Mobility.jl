######################################################################
## Intergenerational Mobility Model (Lee and Seshadri, 2019 JPE) #####
######################################################################
# by Yeonggyu Yun, Stefano Lord-Medrano, and Fernando de Lima Lopes ##
######################################################################

using Parameters, Statistics, Plots, CSV, Tables, Random, Distributions, DataFrames, Interpolations, Optim, JuMP, Ipopt

##### 1. Housekeeping

Random.seed!(1234)

@with_kw struct Params
    T::Int64 = 13 # 13 periods in life
    r::Float64 = (1.04)^6 - 1 # Interst rate
    β::Float64 = (1 / 1.04)^6 # Time discount
    σ::Float64 = 2.0 # Risk aversion
    θ::Float64 = 0.32 # Altruism

    κ::Array{Float64, 1} = [1,1,1,1,1,1,1,1,1] # Age specific income for periods 4-12
    ω_c::Float64 = 0.5
    ρ_h::Float64 = 0.7 # Can choose different values!
    σ_h::Float64 = 0.97^6
    N_η::Int64 = 20

    # Grids for parents human capital h and kids human capital hc

    N_h::Int64 = 30
    N_hc::Int64 = 30
    h_grid::Array{Float64, 1} = range(start = 0.0, stop = 30.0, length = N_h)
    hc_grid::Array{Float64, 1} = range(start = 0.0, stop = 30.0, length = N_hc)

    # Grids for asset b, investment in kids i, transfer to kids τ

    N_b::Int64 = 81
    N_τ::Int64 = 40
    N_i::Int64 = 40
    b_grid::Array{Float64, 1} = range(start = -40.0, stop = 40.0, length = N_b)
    τ_grid::Array{Float64, 1} = range(start = 0.0, stop = 40.0, length = N_τ)
    i_grid::Array{Float64, 1} = range(start = 0.0, stop = 40.0, length = N_i)

    # Simulation

    N_i::Int64 = 10000 # Number of individuals to simulate
    T_sim::Int64 = 30 # Number of periods to simulate
    burnin::Int64 = 45
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
    T::Array{Float64, 3} = zeros(pars.N_b, pars.N_h, pars.N_hc) # Transfer to kids in period 9

    # Discretized grid of z

    ηs::Array{Float64, 1} = zeros(pars.N_η)
    Π_η::Array{Float64, 1} = zeros(pars.N_η, pars.N_η)

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

function trans_Tauchen(pars)
    @unpack σ_h, ρ_h, N_η = pars

    ## Discretize the space of permanent component

    zN = 3*(σ_h^2 / (1-ρ_h^2))^(1/2)
    z1 = -zN

    Π_η = zeros(N_η, N_η)

    ηs = range(start = z1, stop = zN, length = N_η)
    d1 = zs[2] - zs[1]
    for (j, z_j) in enumerate(zs)
        for (k, z_k) in enumerate(zs)
            if k == 1
                Π_η[j,k] = cdf(Normal(0, 1), (z_k+d1/2-ρ_h*z_j)/σ_h)
            elseif k == N_perp
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
    @unpack β, r, σ, T, θ, κ, ω_c, h_grid, hc_grid, b_grid, τ_grid, i_grid = pars

    ## Discretize the space of z following Tauchen
    
    res.ηs, res.Π_η = trans_Tauchen(pars)
    @unpack ηs, Π_η = res

    ## Initialize 

    V= zeros(pars.N_b, 4, pars.N_h) # Value function for periods 4, 10, 11, 12
    Vc = zeros(pars.N_b, 5, pars.N_h, pars.N_hc) # Value function for periods 5, 6, 7, 8, 9
    B = zeros(pars.N_b, 4, pars.N_h) # Asset choice in periods 4, 10, 11, 12
    Bc = zeros(pars.N_b, 5, pars.N_h, pars.N_hc) # Asset choice in peridos 5, 6, 7, 8, 9
    I = zeros(pars.N_b, 4, pars.N_h, pars.N_hc) # Investment choice in periods 5, 6, 7, 8
    Tr = zeros(pars.N_b, pars.N_h, pars.N_hc) # Transfer to kids in period 9

    b_interp = interpolate(b_grid, BSpline(Linear()))
    τ_interp = interpolate(τ_grid, BSpline(Linear()))
    i_interp = interpolate(i_grid, BSpline(Linear()))
    
    ## Periods 10, 11, 12

    for (i_b, b) in enumerate(b_grid)
        for (i_h, h) in enumerate(h_grid)
            B[i_b, 4, i_h] = 0
            c = exp(κ[13] + h) + (1+r) * B
            V[i_b, 4, i_h] = (c^(1-σ) - 1) / (1 - σ)
        end
    end

    for t in range(3:-1:2)
        for (i_b, b) in enumerate(b_grid)
            for (i_h, h) in enumerate(h_grid)

            function
            
            end
        end
    end

    ## Periods 5, 6, 7, 8, 9

    ## Period 4


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
    mean_by_age[i] = mean(sims.E[sims.A .== i])
end

plot(mean_by_age)