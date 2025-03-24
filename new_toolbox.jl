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


#define initial state
function peps_Gauge(A::PEPSTensor, Be::Vector{Float64}, Bo::Vector{Float64})
    p = 1
    v = Int(D / 2)

    PB = Z2Space(0 => 2 * p)
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


#Utility functions for CTMRG
function pancakemaker(A::TensorMap)
    cake = adjoint(A) * A
    cake = permute(cake, (8, 4, 7, 3), (5, 1, 6, 2))
    V = space(cake, 1)
    U = isometry(fuse(V, V), V ⊗ V')

    @tensor pancake[-1 -2; -3 -4] := cake[1 2 3 4; 5 6 7 8] * U[-1; 1 2] * U[-2; 3 4] * adjoint(U)[5 6; -3] * adjoint(U)[7 8; -4]
    return cake
end

function gauge_isometries(B::TensorMap)
    cake = adjoint(B) * B
    @tensor eclair[-3 -4; -1 -2] := cake[1 -2 2 -4; 1 -1 2 -3]
    U, S, V, _ = tsvd(eclair, (1, 2), (3, 4), trunc=truncerr(0.01))
    P1 = U * sqrt(S)
    P2 = sqrt(S) * V
    return P1, P2, U, V, S
end

function partition_function_peps(Ψ::InfinitePEPS)
    P1, P2, _, _, _ = gauge_isometries(Ψ[1, 2])
    squashed_A = pancakemaker(Ψ[1, 1])

    @tensor A_bar[-1 -2; -3 -4] := squashed_A[1 2 3 4; 5 6 7 8] * P2[-1; 1 2] * P2[-2; 3 4] * P1[5 6; -3] * P1[7 8; -4]
    return InfinitePartitionFunction(A_bar)
end

function environment_Z(Z::InfinitePartitionFunction, ctmalg::PEPSKit.CTMRGAlgorithm, χenv::Int)
    env0 = CTMRGEnv(Z, Z2Space(0 => χenv))
    env, = leading_boundary(env0, Z, ctmalg)
    return env
end

function retrieve_old_environment(env::CTMRGEnv, Ψ::InfinitePEPS)

    P1, P2, U1, U2, S = gauge_isometries(Ψ[1, 2])
    S_inv = inv(sqrt(S))

    V = Z2Space(0 => χenv)
    trivial = Z2Space(0 => 1)
    identity_edge_type1 = isometry(V ⊗ trivial ⊗ trivial', V)
    identity_edge_type2 = isometry(V ⊗ trivial' ⊗ trivial, V)

    env_init_peps = CTMRGEnv(Ψ, Z2Space(0 => χenv))

    @tensor env_init_peps.edges[4, 1, 2][-1 -2 -3; -4] := env.edges[4][-1 1; -4] * P2[1; -2 -3]
    @tensor env_init_peps.edges[1, 2, 1][-1 -2 -3; -4] := env.edges[1][-1 1; -4] * P1[-2 -3; 1]
    @tensor env_init_peps.edges[2, 1, 2][-1 -2 -3; -4] := env.edges[2][-1 1; -4] * P1[-2 -3; 1]
    @tensor env_init_peps.edges[3, 2, 1][-1 -2 -3; -4] := env.edges[3][-1 1; -4] * P2[1; -2 -3]

    @tensor env_init_peps.edges[4, 1, 1][-1 -2 -3; -4] := env.edges[4][-1 1; -4] * S_inv[1; 2] * adjoint(U1)[2; -2 -3]
    @tensor env_init_peps.edges[2, 1, 1][-1 -2 -3; -4] := env.edges[2][-1 1; -4] * S_inv[2; 1] * adjoint(U2)[-2 -3; 2]
    env_init_peps.edges[1, 2, 2] = identity_edge_type1
    env_init_peps.edges[3, 2, 2] = identity_edge_type2


    @tensor env_init_peps.edges[1, 1, 1][-1 -2 -3; -4] := env.edges[1][-1 1; -4] * S_inv[2; 1] * adjoint(U2)[-2 -3; 2]
    @tensor env_init_peps.edges[3, 1, 1][-1 -2 -3; -4] := env.edges[3][-1 1; -4] * S_inv[1; 2] * adjoint(U1)[2; -2 -3]
    env_init_peps.edges[2, 2, 2] = identity_edge_type1
    env_init_peps.edges[4, 2, 2] = identity_edge_type2

    env_init_peps.edges[1, 1, 2] = identity_edge_type1
    env_init_peps.edges[3, 1, 2] = identity_edge_type2
    env_init_peps.edges[2, 2, 1] = identity_edge_type1
    env_init_peps.edges[4, 2, 1] = identity_edge_type2

    env_init_peps.corners[1, 2, 2] = env.corners[1]
    env_init_peps.corners[3, 2, 2] = env.corners[3]
    env_init_peps.corners[2, 2, 2] = env.corners[2]
    env_init_peps.corners[4, 2, 2] = env.corners[4]

    env_init_peps.corners[1, 2, 1] = env.corners[1]
    env_init_peps.corners[3, 2, 1] = env.corners[3]
    env_init_peps.corners[2, 2, 1] = env.corners[2]
    env_init_peps.corners[4, 2, 1] = env.corners[4]

    env_init_peps.corners[1, 1, 2] = env.corners[1]
    env_init_peps.corners[3, 1, 2] = env.corners[3]
    env_init_peps.corners[2, 1, 2] = env.corners[2]
    env_init_peps.corners[4, 1, 2] = env.corners[4]

    env_init_peps.corners[1, 1, 1] = env.corners[1]
    env_init_peps.corners[3, 1, 1] = env.corners[3]
    env_init_peps.corners[2, 1, 1] = env.corners[2]
    env_init_peps.corners[4, 1, 1] = env.corners[4]

    return env_init_peps
end

function new_leading_boundary(Ψ::InfinitePEPS, ctmalg::PEPSKit.CTMRGAlgorithm, χenv::Int)
    Z = partition_function_peps(Ψ)
    env = environment_Z(Z, ctmalg, χenv)
    return retrieve_old_environment(env, Ψ)
end

function _rrule(
    gradmode::GradMode{:diffgauge},
    config::RuleConfig,
    ::typeof(new_leading_boundary),
    state,
    alg::CTMRGAlgorithm,
)
    env = new_leading_boundary(state, alg, χ_env)
    alg_fixed = @set alg.projector_alg.trscheme = FixedSpaceTruncation() # fix spaces during differentiation

    function new_leading_boundary_diffgauge_pullback((Δenv′, Δinfo))
        Δenv = unthunk(Δenv′)

        # find partial gradients of gauge_fixed single CTMRG iteration
        function f(A, x)
            return gauge_fix(x, ctmrg_iteration(InfiniteSquareNetwork(A), x, alg_fixed)[1])[1]
        end
        _, env_vjp = rrule_via_ad(config, f, state, env)

        # evaluate the geometric sum
        ∂f∂A(x)::typeof(state) = env_vjp(x)[2]
        ∂f∂x(x)::typeof(env) = env_vjp(x)[3]
        ∂F∂env = fpgrad(Δenv, ∂f∂x, ∂f∂A, Δenv, gradmode)

        return NoTangent(), ZeroTangent(), ∂F∂env, NoTangent()
    end

    return (env, info), leading_boundary_diffgauge_pullback
end


#Define Hamiltonian
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

function costfun(Ψ, envs, H)
    E = MPSKit.expectation_value(Ψ, H, envs)
    return real(E)
end
function update!(env::CTMRGEnv{C,T}, env´::CTMRGEnv{C,T}) where {C,T}
    env.corners .= env´.corners
    env.edges .= env´.edges
    return env
end
reuse_env = true
