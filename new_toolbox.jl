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

function peps_Gauge(A::PEPSTensor, Be::Vector{ComplexF64}, Bo::Vector{ComplexF64})
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

function peps_Gauge(A::PEPSTensor, Be::Matrix{ComplexF64}, Bo::Matrix{ComplexF64})
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

    return InfinitePEPS([A BW; BD T])
end


function peps_Gauge(A::PEPSTensor, Be::Matrix{Float64}, Bo::Matrix{Float64})
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

    return InfinitePEPS([A BW; BD T])
end

function peps_Gauge_trivial(A::PEPSTensor)
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
            BW[s, f][1, 1, :, 1, :] = Matrix{ComplexF64}(I, v, v)
        else
            BW[s, f][2, 1, :, 1, :] = Matrix{ComplexF64}(I, v, v)
        end
    end

    for (s, f) in fusiontrees(BD)
        if f.uncoupled[1] == Irrep[ℤ₂](0)
            BD[s, f][1, :, 1, :, 1] = Matrix{ComplexF64}(I, v, v)
        else
            BD[s, f][2, :, 1, :, 1] = Matrix{ComplexF64}(I, v, v)
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

        return NoTangent(), dA, collect(real(dBe)), collect(real(dBo))
    end
    return Ψ, peps_Gauge_pullback
end

function ChainRulesCore.rrule(::typeof(peps_Gauge), A::PEPSTensor, Be::Vector{ComplexF64}, Bo::Vector{ComplexF64})
    Ψ = peps_Gauge(A, Be, Bo)

    function peps_Gauge_pullback(dΨ)
        dA = dΨ[1, 1]
        dBW = dΨ[1, 2]

        trees = collect(fusiontrees(dBW))
        te = only(filter(((s, f),) -> f.uncoupled[2] == Irrep[ℤ₂](0), trees))
        to = only(filter(((s, f),) -> f.uncoupled[2] == Irrep[ℤ₂](1), trees))
        dBe = diag(dBW[te[1], te[2]][1, 1, :, 1, :])
        dBo = diag(dBW[to[1], to[2]][2, 1, :, 1, :])

        return NoTangent(), dA, collect(dBe), collect(dBo)
    end
    return Ψ, peps_Gauge_pullback
end

function ChainRulesCore.rrule(::typeof(peps_Gauge), A::PEPSTensor, Be::Matrix{ComplexF64}, Bo::Matrix{ComplexF64})
    Ψ = peps_Gauge(A, Be, Bo)

    function peps_Gauge_pullback(dΨ)
        dA = dΨ[1, 1]
        dBW = dΨ[1, 2]

        trees = collect(fusiontrees(dBW))
        te = only(filter(((s, f),) -> f.uncoupled[2] == Irrep[ℤ₂](0), trees))
        to = only(filter(((s, f),) -> f.uncoupled[2] == Irrep[ℤ₂](1), trees))
        dBe = dBW[te[1], te[2]][1, 1, :, 1, :]
        dBo = dBW[to[1], to[2]][2, 1, :, 1, :]

        return NoTangent(), dA, collect(dBe), collect(dBo)
    end
    return Ψ, peps_Gauge_pullback
end

function ChainRulesCore.rrule(::typeof(peps_Gauge), A::PEPSTensor, Be::Matrix{Float64}, Bo::Matrix{Float64})
    Ψ = peps_Gauge(A, Be, Bo)

    function peps_Gauge_pullback(dΨ)
        dA = dΨ[1, 1]
        dBW = dΨ[1, 2]

        trees = collect(fusiontrees(dBW))
        te = only(filter(((s, f),) -> f.uncoupled[2] == Irrep[ℤ₂](0), trees))
        to = only(filter(((s, f),) -> f.uncoupled[2] == Irrep[ℤ₂](1), trees))
        dBe = dBW[te[1], te[2]][1, 1, :, 1, :]
        dBo = dBW[to[1], to[2]][2, 1, :, 1, :]

        return NoTangent(), dA, collect(dBe), collect(dBo)
    end
    return Ψ, peps_Gauge_pullback
end

function ChainRulesCore.rrule(::typeof(peps_Gauge_trivial), A::PEPSTensor)
    Ψ = peps_Gauge_trivial(A)

    function peps_Gauge_trivial_pullback(dΨ)
        dA = dΨ[1, 1]
        return NoTangent(), dA
    end
    return Ψ, peps_Gauge_trivial_pullback
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


    # Z_buffer = InfinitePartitionFunction(TensorMap(zeros, ComplexF64, space(P2, 1) ⊗ space(P2, 1) ← space(P1, 3)' ⊗ space(P1, 3)'))
    # A_bar = Zygote.Buffer(Z_buffer.A)
    @tensor A_bar[-1 -2; -3 -4] := squashed_A[1 2 3 4; 5 6 7 8] * P2[-1; 1 2] * P2[-2; 3 4] * P1[5 6; -3] * P1[7 8; -4]
    m = Zygote.Buffer(Matrix{typeof(A_bar)}(undef, 1, 1))
    m[1, 1] = A_bar
    return InfiniteSquareNetwork(copy(m))
end


function ChainRulesCore.rrule(::Type{<:InfinitePartitionFunction}, A::Matrix{<:AbstractTensorMap{ComplexF64,S,2,2} where {S<:ElementarySpace}})
    Z = InfinitePartitionFunction(A)

    function InfinitePartitionFunction_pullback(dZ)
        return NoTangent(), PEPSKit.unitcell(dZ)
    end
    return Z, InfinitePartitionFunction_pullback
end

function ChainRulesCore.rrule(::Type{<:InfiniteSquareNetwork}, A::Matrix{<:AbstractTensorMap{ComplexF64,S,2,2} where {S<:ElementarySpace}})
    Z = InfiniteSquareNetwork(A)

    function InfiniteSquareNetwork_pullback(dZ)
        return NoTangent(), PEPSKit.unitcell(dZ)
    end
    return Z, InfiniteSquareNetwork_pullback
end




function get_new_environment_Z(env::CTMRGEnv, Ψ::InfinitePEPS)
    P1, P2, U1, U2, S = gauge_isometries(Ψ[1, 2])
    S_inv = inv(sqrt(S))
    Z = partition_function_peps(Ψ)
    χenv = space(env.edges[1], 1)
    env_Z_buffer = CTMRGEnv(Z, χenv)
    corners = Zygote.Buffer(env_Z_buffer.corners)
    edges = Zygote.Buffer(env_Z_buffer.edges)
    @tensor edges[4][-1 -2; -3] := env.edges[4, 1, 2][-1 1 2; -3] * adjoint(U2)[1 2; 3] * S_inv[3; -2]
    @tensor edges[1][-1 -2; -3] := env.edges[1, 2, 1][-1 1 2; -3] * adjoint(U1)[3; 1 2] * S_inv[-2; 3]
    @tensor edges[2][-1 -2; -3] := env.edges[2, 1, 2][-1 1 2; -3] * adjoint(U1)[3; 1 2] * S_inv[-2; 3]
    @tensor edges[3][-1 -2; -3] := env.edges[3, 2, 1][-1 1 2; -3] * adjoint(U2)[1 2; 3] * S_inv[3; -2]

    corners[1] = env.corners[1, 2, 2]
    corners[3] = env.corners[3, 2, 2]
    corners[2] = env.corners[2, 2, 2]
    corners[4] = env.corners[4, 2, 2]

    return CTMRGEnv(copy(corners), copy(edges))

end

# function environment_Z(env0::CTMRGEnv, Z::InfinitePartitionFunction, ctmalg::PEPSKit.CTMRGAlgorithm)
#     env, = leading_boundary(env0, Z, ctmalg)
#     return env
# end

function retrieve_old_environment(env::CTMRGEnv, Ψ::InfinitePEPS)

    P1, P2, U1, U2, S = gauge_isometries(Ψ[1, 2])
    S_inv = inv(sqrt(S))
    χenv = space(env.edges[1], 1)
    V = χenv
    trivial = Z2Space(0 => 1)
    identity_edge_type1 = isometry(V ⊗ trivial ⊗ trivial', V)
    identity_edge_type2 = isometry(V ⊗ trivial' ⊗ trivial, V)

    env_init_peps_buffer = CTMRGEnv(Ψ, χenv)
    edges = Zygote.Buffer(env_init_peps_buffer.edges)
    corners = Zygote.Buffer(env_init_peps_buffer.corners)

    @tensor edges[4, 1, 2][-1 -2 -3; -4] := env.edges[4][-1 1; -4] * P2[1; -2 -3]
    @tensor edges[1, 2, 1][-1 -2 -3; -4] := env.edges[1][-1 1; -4] * P1[-2 -3; 1]
    @tensor edges[2, 1, 2][-1 -2 -3; -4] := env.edges[2][-1 1; -4] * P1[-2 -3; 1]
    @tensor edges[3, 2, 1][-1 -2 -3; -4] := env.edges[3][-1 1; -4] * P2[1; -2 -3]

    @tensor edges[4, 1, 1][-1 -2 -3; -4] := env.edges[4][-1 1; -4] * S_inv[1; 2] * adjoint(U1)[2; -2 -3]
    @tensor edges[2, 1, 1][-1 -2 -3; -4] := env.edges[2][-1 1; -4] * S_inv[2; 1] * adjoint(U2)[-2 -3; 2]
    edges[1, 2, 2] = identity_edge_type1
    edges[3, 2, 2] = identity_edge_type2


    @tensor edges[1, 1, 1][-1 -2 -3; -4] := env.edges[1][-1 1; -4] * S_inv[2; 1] * adjoint(U2)[-2 -3; 2]
    @tensor edges[3, 1, 1][-1 -2 -3; -4] := env.edges[3][-1 1; -4] * S_inv[1; 2] * adjoint(U1)[2; -2 -3]
    edges[2, 2, 2] = identity_edge_type1
    edges[4, 2, 2] = identity_edge_type2

    edges[1, 1, 2] = identity_edge_type1
    edges[3, 1, 2] = identity_edge_type2
    edges[2, 2, 1] = identity_edge_type1
    edges[4, 2, 1] = identity_edge_type2

    corners[1, 2, 2] = env.corners[1]
    corners[3, 2, 2] = env.corners[3]
    corners[2, 2, 2] = env.corners[2]
    corners[4, 2, 2] = env.corners[4]

    corners[1, 2, 1] = env.corners[1]
    corners[3, 2, 1] = env.corners[3]
    corners[2, 2, 1] = env.corners[2]
    corners[4, 2, 1] = env.corners[4]

    corners[1, 1, 2] = env.corners[1]
    corners[3, 1, 2] = env.corners[3]
    corners[2, 1, 2] = env.corners[2]
    corners[4, 1, 2] = env.corners[4]

    corners[1, 1, 1] = env.corners[1]
    corners[3, 1, 1] = env.corners[3]
    corners[2, 1, 1] = env.corners[2]
    corners[4, 1, 1] = env.corners[4]

    return CTMRGEnv(copy(corners), copy(edges))
end


# function ChainRulesCore.rrule(::typeof(retrieve_old_environment), env::CTMRGEnv, Ψ::InfinitePEPS)
#     env_final = retrieve_old_environment(env, Ψ)
#     Z = partition_function_peps(Ψ)
#     P1, P2, U1, U2, S = gauge_isometries(Ψ[1, 2])
#     S_inv = inv(sqrt(S))
#     χenv = space(env.edges[1], 1)
#     function retrieve_old_environment_pullback(denv_final)
#         denv = CTMRGEnv(Z, χenv)

#         @tensor denv.edges[4][-1 -2; -3] := denv_final.edges[4, 1, 2][-1 1 2; -3] * adjoint(U2)[1 2; 3] * S_inv[3; -2]
#         @tensor denv.edges[1][-1 -2; -3] := denv_final.edges[1, 2, 1][-1 1 2; -3] * adjoint(U1)[3; 1 2] * S_inv[-2; 3]
#         @tensor denv.edges[2][-1 -2; -3] := denv_final.edges[2, 1, 2][-1 1 2; -3] * adjoint(U1)[3; 1 2] * S_inv[-2; 3]
#         @tensor denv.edges[3][-1 -2; -3] := denv_final.edges[3, 2, 1][-1 1 2; -3] * adjoint(U2)[1 2; 3] * S_inv[3; -2]

#         denv.corners[1] = denv_final.corners[1, 2, 2]
#         denv.corners[3] = denv_final.corners[3, 2, 2]
#         denv.corners[2] = denv_final.corners[2, 2, 2]
#         denv.corners[4] = denv_final.corners[4, 2, 2]


#         return NoTangent(), denv
#     end
#     return env_final, retrieve_old_environment_pullback
# end

function new_leading_boundary(env::CTMRGEnv, Ψ::InfinitePEPS, ctmalg::PEPSKit.CTMRGAlgorithm)
    Z = partition_function_peps(Ψ)
    env0 = get_new_environment_Z(env, Ψ)
    env_Z, = leading_boundary(env0, Z, ctmalg)
    env_final = retrieve_old_environment(env_Z, Ψ)
    return env_final
end

# function _rrule(
#     gradmode::GradMode{:diffgauge},
#     config::RuleConfig,
#     ::typeof(new_leading_boundary),
#     state,
#     alg::CTMRGAlgorithm,
# )
#     env = new_leading_boundary(state, alg, χ_env)
#     alg_fixed = @set alg.projector_alg.trscheme = FixedSpaceTruncation() # fix spaces during differentiation

#     function new_leading_boundary_diffgauge_pullback((Δenv′, Δinfo))
#         Δenv = unthunk(Δenv′)

#         # find partial gradients of gauge_fixed single CTMRG iteration
#         function f(A, x)
#             return gauge_fix(x, ctmrg_iteration(InfiniteSquareNetwork(A), x, alg_fixed)[1])[1]
#         end
#         _, env_vjp = rrule_via_ad(config, f, state, env)

#         # evaluate the geometric sum
#         ∂f∂A(x)::typeof(state) = env_vjp(x)[2]
#         ∂f∂x(x)::typeof(env) = env_vjp(x)[3]
#         ∂F∂env = fpgrad(Δenv, ∂f∂x, ∂f∂A, Δenv, gradmode)

#         return NoTangent(), ZeroTangent(), ∂F∂env, NoTangent()
#     end

#     return (env, info), leading_boundary_diffgauge_pullback
# end


#Define Hamiltonian
function Fradkin_Shenker(lattice::InfiniteSquare; kwargs...)
    return Fradkin_Shenker(ComplexF64, Z2Irrep, lattice; kwargs...)
end

function Fradkin_Shenker(T::Type{<:Number},
    S::Type{<:Sector},
    lattice::InfiniteSquare; Jx::Number, Jz::Number, hx::Number, hz::Number, pdim::Int, vdim::Int)
    p = Int(pdim / 2)
    v = Int(vdim / 2)

    PB = Z2Space(0 => 2 * p)
    PA = Z2Space(0 => p, 1 => p)
    PT = Z2Space(0 => p)
    V = Z2Space(0 => v, 1 => v)
    II = Z2Space(0 => 1)

    Z = TensorMap(zeros, ComplexF64, PA ← PA)
    GZ = TensorMap(ComplexF64[1.0 0.0; 0.0 -1.0], PB ← PB)
    GX = TensorMap(ComplexF64[0.0 1.0; 1.0 0.0], PB ← PB)
    XX = TensorMap(zeros, ComplexF64, PA ⊗ PA ← PA ⊗ PA)
    @tensor PLAQ[-1 -2 -3 -4; -5 -6 -7 -8] := GX[-1; -5] * GX[-2; -6] * GX[-3; -7] * GX[-4; -8]

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
        ((matter,) => -Jx * Z for matter in matter_sites)...,
        (plaq => -Jz * PLAQ for plaq in plaqs)...,
        ((gauge,) => -hx * GZ for gauge in gauge_sites)...,
        (bond => -hz * B for bond in bonds)...,)

end

function plaq(lattice::InfiniteSquare)
    neighbors = Tuple{CartesianIndex,CartesianIndex,CartesianIndex,CartesianIndex}[]
    for idx in PEPSKit.vertices(lattice)
        if isodd(idx[1]) && isodd(idx[2])
            push!(neighbors, (idx + CartesianIndex(0, 1), idx + CartesianIndex(1, 2), idx + CartesianIndex(2, 1), idx + CartesianIndex(1, 0)))
        end
    end
    return neighbors
end

function triplebond(lattice::InfiniteSquare)
    neighbors = Tuple{CartesianIndex,CartesianIndex,CartesianIndex}[]
    for idx in PEPSKit.vertices(lattice)
        if isodd(idx[1]) && isodd(idx[2])
            push!(neighbors, (idx, idx + CartesianIndex(0, 1), idx + CartesianIndex(0, 2)))
            push!(neighbors, (idx, idx + CartesianIndex(1, 0), idx + CartesianIndex(2, 0)))
        end
    end
    return neighbors
end
function my_retract_old(x, dx, α)
    A, Be, Bo, env = deepcopy(x)
    dA, dBe, dBo = dx
    A += α * dA
    Be += α * dBe
    Bo += α * dBo

    # env = leading_boundary(env, peps_Gauge(A, Be, Bo), ctm_alg)

    return (A, Be, Bo, env), dx
end

my_symmetrize(A::PEPSKit.PEPSTensor) = PEPSKit.herm_depth_inv(PEPSKit.rot_inv(A))

function my_symmetrize(x::Tuple{PEPSKit.PEPSTensor,Vector{Float64},Vector{Float64}})
    A, Be, Bo = x
    A = my_symmetrize(A)

    return (A, Be, Bo)
end

function my_symmetrize(x::Tuple{PEPSKit.PEPSTensor})
    A, = x
    A = my_symmetrize(A)

    return (A,)
end


function my_retract(x, dx, α)
    A, Be, Bo, env = deepcopy(x)
    dA, dBe, dBo = dx


    A, dxA = PEPSKit.norm_preserving_retract(A, dA, α)
    Be, dxBe = PEPSKit.norm_preserving_retract(Be, dBe, α)
    Bo, dxBo = PEPSKit.norm_preserving_retract(Bo, dBo, α)

    A = my_symmetrize(A)
    dxA = my_symmetrize(dxA)
    # env = leading_boundary(env, peps_Gauge(A, Be, Bo), ctm_alg)


    return (A, Be, Bo, env), (dxA, dxBe, dxBo)
end

function my_transport!(ξ, x, dx, α, A´)
    A, Be, Bo, env = deepcopy(x)
    dA, dBe, dBo = dx
    ξA, ξBe, ξBo = ξ


    ξA = PEPSKit.norm_preserving_transport!(ξA, A, dA, α, A´)
    ξBe = PEPSKit.norm_preserving_transport!(ξBe, Be, dBe, α, A´)
    ξBo = PEPSKit.norm_preserving_transport!(ξBo, Bo, dBo, α, A´)

    return (ξA, ξBe, ξBo)
end

function my_retract_trivial(x, dx, α)
    A, env = deepcopy(x)
    dA, = dx

    #A += α * dA
    A, dxA = PEPSKit.norm_preserving_retract(A, dA, α)
    A = my_symmetrize(A)
    dxA = my_symmetrize(dxA)
    # env = leading_boundary(env, peps_Gauge(A, Be, Bo), ctm_alg)

    return (A, env), (dxA,)
end

function my_transport_trivial!(ξ, x, dx, α, A´)
    A, env = deepcopy(x)
    dA, = dx
    ξA, = ξ

    ξA = PEPSKit.norm_preserving_transport!(ξA, A, dA, α, A´)

    return (ξA,)
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
    return real(dot(v1[1], v2[1])) + real(dot(v1[2], v2[2])) + real(dot(v1[3], v2[3]))
end

function my_inner_trivial(x, v1, v2)
    return real(dot(v1[1], v2[1]))
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
