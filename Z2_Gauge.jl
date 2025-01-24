using LinearAlgebra
using TensorKit
using MPSKit
using PEPSKit
using Base.Iterators
using OptimKit
using KrylovKit
using VectorInterface
using MPSKitModels

using Zygote
using ChainRulesCore
using Dates
using JLD2
using PEPSKit: PEPSTensor, CTMRGEnv, NORTH, SOUTH, WEST, EAST, NORTHWEST, NORTHEAST, SOUTHEAST, SOUTHWEST, _prev, _next, GradMode


# initialize states


function Fradkin_Shenker(lattice::InfiniteSquare; kwargs...)
    return Fradkin_Shenker(ComplexF64, Z2Irrep, lattice; kwargs...)    
end

function Fradkin_Shenker(T::Type{<:Number},
    S::Type{<:Sector},
    lattice::InfiniteSquare; Jx::Number, Jz::Number, hx::Number, hz::Number, pdim::Int, vdim::Int)
    p = Int(pdim / 2)
    v = Int(vdim / 2)

    PB = Z2Space(0 => 2*p)
    PA = Z2Space(0 => p, 1 => p)
    PT = Z2Space(0 => p)
    V = Z2Space(0 => v, 1 => v)
    II = Z2Space(0 => 1)

    Z = TensorMap(zeros, ComplexF64, PA ← PA)
    GZ = TensorMap(ComplexF64[1.0 0.0; 0.0 -1.0], PB ← PB)
    GX = TensorMap(ComplexF64[0.0 1.0; 1.0 0.0], PB ← PB)
    XX = TensorMap(zeros, ComplexF64, PA ⊗ PA ← PA ⊗ PA)
    @tensor PLAQ[-1 -2 -3 -4; -5 -6 -7 -8] := GX[-1, -5] * GX[-2, -6] * GX[-3, -7] * GX[-4, -8]

    for (s, f) in fusiontrees(Z)
        if s.uncoupled[1] == Irrep[ℤ₂](0)
            Z[s, f][1] = 1
        else
            Z[s, f][1] = -1
        end
    end

    for (s, f) in fusiontrees(XX)
        if only(s.uncoupled[1] ⊗ f.uncoupled[1]) == Z2Irrep(1)
            XX[s, f] .= 1
        end
    end

    @tensor B[-1 -2 -3; -4 -5 -6] := XX[-1 -3; -4 -6] * GX[-2; -5]

    

    spaces = Matrix{Z2Space}(undef, lattice.Nrows, lattice.Ncols)
    lattice = InfiniteSquare(2, 2)
    for i = 1:lattice.Nrows
        for j = 1:lattice.Ncols
            if isodd(i) && isodd(j)
                spaces[i, j] = PA
         
            elseif iseven(i) && iseven(j)
                spaces[i, j] = PT
        
            else    
                spaces[i, j] = PB
            end
        end
    end

    bonds = triplebond(lattice)
    plaqs = plaq(lattice)

    matter_sites = filter(idx -> isodd(idx[1]) && isodd(idx[2]), PEPSKit.vertices(lattice))
    gauge_sites = filter(idx -> (isodd(idx[1]) && iseven(idx[2])) || (iseven(idx[1]) && isodd(idx[2])), PEPSKit.vertices(lattice))
    return PEPSKit.LocalOperator(
        spaces,
        (plaq => -Jz * PLAQ for plaq in plaqs)...,
        (bond => -hz * B for bond in bonds)...,
        ((matter,) => -Jx * Z for matter in matter_sites)...,
        ((gauge,) => -hx * GZ for gauge in gauge_sites)...,
    )

end

function plaq(lattice::InfiniteSquare)
    neighbors = Tuple{CartesianIndex,CartesianIndex, CartesianIndex, CartesianIndex}[]
    for idx in PEPSKit.vertices(lattice)
        if isodd(idx[1]) && isodd(idx[2])
            push!(neighbors, (idx + CartesianIndex(0,1), idx + CartesianIndex(1,2), idx + CartesianIndex(2,1), idx + CartesianIndex(1,0)))
        end
    end
    return neighbors
end

function triplebond(lattice::InfiniteSquare)
    neighbors = Tuple{CartesianIndex, CartesianIndex, CartesianIndex}[]
    for idx in PEPSKit.vertices(lattice)
        if isodd(idx[1]) && isodd(idx[2])
            push!(neighbors, (idx, idx + CartesianIndex(0,1), idx + CartesianIndex(0,2)))
            push!(neighbors, (idx, idx + CartesianIndex(1,0), idx + CartesianIndex(2,0)))
        end
    end
    return neighbors
end

function gauge_inv_peps(pdim::Int, vdim::Int, ::Type{Z2Irrep})
    p = Int(pdim / 2)
    v = Int(vdim / 2)

    PB = Z2Space(0 => 2*p)
    PA = Z2Space(0 => p, 1 => p)
    V = Z2Space(0 => v, 1 => v)
    II = Z2Space(0 => 1)

    A = TensorMap(randn, ComplexF64, PA ← V ⊗ V ⊗ V' ⊗ V')
    BW = TensorMap(zeros, ComplexF64, PB ← II ⊗ V ⊗ (II)' ⊗ (V)')
    BD = TensorMap(zeros, ComplexF64, PB ← V ⊗ II ⊗ (V)' ⊗ (II)')
    T = TensorMap(ones, ComplexF64, II ← II ⊗ II ⊗ (II)' ⊗ (II)')

    Be = diagm(diag(rand(ComplexF64,v,v)))
    Bo = diagm(diag(rand(ComplexF64,v,v)))

    for (s, f) in fusiontrees(BW)
        if f.uncoupled[2] == Irrep[ℤ₂](0)
            BW[s, f][1, 1, :, 1, :] = Be
        else
            BW[s, f][2, 1, :, 1, :] = Bo
        end
    end

    for (s, f) in fusiontrees(BD)
        if f.uncoupled[1] == Irrep[ℤ₂](0)
            BD[s, f][1, :, 1, :, 1] = Be
        else
            BD[s, f][2, :, 1, :, 1] = Bo
        end
    end

    return InfinitePEPS([A BW; BD T]), A, Be, Bo
end

function peps_Gauge(A::PEPSTensor, Be::Vector{Float64}, Bo::Vector{Float64})
    p = 1
    v = Int(D / 2)

    PB = Z2Space(0 => 2*p)
    V = Z2Space(0 => v, 1 => v)
    II = Z2Space(0 => 1)

    BW = TensorMap(zeros, ComplexF64, PB ← II ⊗ V ⊗ (II)' ⊗ (V)')
    BD = TensorMap(zeros, ComplexF64, PB ← V ⊗ II ⊗ (V)' ⊗ (II)')
    T = TensorMap(ones, ComplexF64, II ← II ⊗ II ⊗ (II)' ⊗ (II)')

    for (s, f) in fusiontrees(BW)
        if f.uncoupled[2] == Irrep[ℤ₂](0)
            BW[s, f][1, 1, :, 1, :] = diagm(Be)
        else
            BW[s, f][2, 1, :, 1, :] = diagm(Bo)
        end
    end

    for (s, f) in fusiontrees(BD)
        if f.uncoupled[1] == Irrep[ℤ₂](0)
            BD[s, f][1, :, 1, :, 1] = diagm(Be)
        else
            BD[s, f][2, :, 1, :, 1] = diagm(Bo)
        end
    end

    return InfinitePEPS([A BW; BD T])
end

function ChainRulesCore.rrule(::typeof(peps_Gauge), A::PEPSTensor, Be::Vector{Float64}, Bo::Vector{Float64})
    Ψ = peps_Gauge(A, Be, Bo)

    function peps_Gauge_pullback(dΨ)
        dA = dΨ[1, 1]
        dBW = dΨ[1, 2]

        trees = collect(fusiontrees(dBW))
        te = only(filter(((s, f),) -> f.uncoupled[2] == Irrep[ℤ₂](0), trees))
        to = only(filter(((s, f),) -> f.uncoupled[2] == Irrep[ℤ₂](1), trees))
        dBe = diag(dBW[te[1], te[2]][1, 1, :, 1, :])
        dBo = diag(dBW[to[1], to[2]][2, 1, :, 1, :])    

        return NoTangent(), dA, real(dBe), real(dBo)
    end
    return Ψ, peps_Gauge_pullback
end


χ = 16 # environment bond dimension
D = 4 # PEPS bond dimension
P = 2 # PEPS physical dimension
p = P/2
v = Int(D / 2)
symm = Z2Irrep

H = Fradkin_Shenker(InfiniteSquare(2,2); Jx=1, Jz=0, hx=1, hz=0, pdim=2, vdim=4)

PA = Z2Space(0 => p, 1 => p)
V = Z2Space(0 => v, 1 => v)
A = TensorMap(randn, ComplexF64, PA ← V ⊗ V ⊗ V' ⊗ V')
Be = diag(rand(Float64,v,v))
Bo = diag(rand(Float64,v,v))

Ψ = peps_Gauge(A, Be, Bo)
ctm_alg = CTMRG(verbosity = 4)
opt_alg = PEPSOptimize(;
    boundary_alg=ctm_alg,
    optimizer=LBFGS(4; gradtol=1e-3, verbosity=2),
    gradient_alg=LinSolver(; iterscheme=:diffgauge),
)

env_init = leading_boundary(CTMRGEnv(Ψ, Z2Space(0 => χ)), Ψ, ctm_alg);

# optimize gismos
function my_retract(x, dx, α)
    A, Be, Bo, env = deepcopy(x)
    dA, dBe, dBo = dx

    A += α * dA
    Be += α * dBe
    Bo += α * dBo

    # env = leading_boundary(env, peps_Gauge(A, Be, Bo), ctm_alg)

    return (A, Be, Bo, env), dx
end

function my_scale!(v, α)
    LinearAlgebra.rmul!.(v, α)
    return v
end

function my_add!(vdst, vsrc, α)
    LinearAlgebra.axpy!.(α, vsrc, vdst)
    return vdst
end

function my_inner(x, v1, v2)
    return real(dot(v1[1], v2[1])) + dot(v1[2], v2[2]) + dot(v1[3], v2[3])
end




real_inner(_, η₁, η₂) = real(dot(η₁, η₂))

function hook_pullback(@nospecialize(f), args...; alg_rrule=nothing, kwargs...)
    return f(args...; kwargs...)
end

# function costfun(Ψ, envs)
#     E = MPSKit.expectation_value(Ψ, H, envs)
#     return real(E)
# end
function update!(env::CTMRGEnv{C,T}, env´::CTMRGEnv{C,T}) where {C,T}
    env.corners .= env´.corners
    env.edges .= env´.edges
    return env
end
reuse_env = true


(A, Be, Bo, env), E, ∂E, numfg, convhistory = optimize(
        (A, Be, Bo, env_init), opt_alg.optimizer; retract = my_retract, inner=my_inner, scale! = my_scale!, add! = my_add!, finalize! = OptimKit._finalize!
    ) do (A, Be, Bo, envs)
        E, gs = withgradient(A, Be, Bo) do A, Be, Bo
            Ψ = peps_Gauge(A, Be, Bo)
            envs´ = hook_pullback(
                leading_boundary,
                envs,
                Ψ,
                opt_alg.boundary_alg;
                alg_rrule=opt_alg.gradient_alg,
            )
            ignore_derivatives() do
                opt_alg.reuse_env && update!(envs, envs´)
            end
            return costfun(Ψ, envs, H)
        end
        gs
        return E, gs
    end

# function cfun(x)
#     (As, Bes, Bos, envs) = x

#     function costfun(A, Be, Bo)
#         Ψ .= peps_Gauge(A, Be, Bo)
#         envs = leading_boundary(x[4], Ψ, ctm_alg)
#         E = MPSKit.expectation_value(Ψ, H, envs)
#         ignore_derivatives() do
#             reuse_env && update!(envs, envs)
#         end
#         return real(E)
#     end
    
#     E, g = withgradient(costfun, As, Bes, Bos)
#     ∂E∂A = g[1]

#     # @assert !isnan(norm(∂E∂v))
#     return E, ∂E∂A
# end
new_Ψ = peps_Gauge(A, Be, Bo)


file = jldopen("final_Psi.jld2", "w")
file["Ψ"] = new_Ψ
file["env"] = env
close(file)

Ψ = peps_Gauge(A, Be, Bo)
BW = Ψ[1, 2]
trees = collect(fusiontrees(BW))
te = only(filter(((s, f),) -> (f.uncoupled[2] == Irrep[ℤ₂](0)), trees))

map(trees) do (s, f)
    f.uncoupled[2] == Irrep[ℤ₂](0)
end