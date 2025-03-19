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

#utilities to define PEPS

function gauge_inv_peps(pdim::Int, vdim::Int, ::Type{Z2Irrep})
    p = Int(pdim / 2)
    v = Int(vdim / 2)

    PB = Z2Space(0 => 2 * p)
    PA = Z2Space(0 => p, 1 => p)
    V = Z2Space(0 => v, 1 => v)
    II = Z2Space(0 => 1)

    A = TensorMap(randn, ComplexF64, PA ← V ⊗ V ⊗ V' ⊗ V')
    BW = TensorMap(zeros, ComplexF64, PB ← II ⊗ V ⊗ (II)' ⊗ (V)')
    BD = TensorMap(zeros, ComplexF64, PB ← V ⊗ II ⊗ (V)' ⊗ (II)')
    T = TensorMap(ones, ComplexF64, II ← II ⊗ II ⊗ (II)' ⊗ (II)')

    Be = diagm(diag(rand(ComplexF64, v, v)))
    Bo = diagm(diag(rand(ComplexF64, v, v)))

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


χ = 16 # environment bond dimension
D = 4 # PEPS bond dimension
P = 2 # PEPS physical dimension
p = P / 2
v = Int(D / 2)
symm = Z2Irrep

PA = Z2Space(0 => p, 1 => p)
V = Z2Space(0 => v, 1 => v)
A = TensorMap(randn, ComplexF64, PA ← V ⊗ V ⊗ V' ⊗ V')
Be = diag(rand(Float64, v, v))
Bo = diag(rand(Float64, v, v))

Ψ = peps_Gauge(A, Be, Bo)

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

Z = partition_function_peps(Ψ)
ctm_alg = SequentialCTMRG(; maxiter=150, verbosity=2)
χenv = 24
env0 = CTMRGEnv(Z, Z2Space(0 => χenv))
env = leading_boundary(env0, Z, ctm_alg);

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

#env_peps = leading_boundary(env_init_peps, Ψ, ctm_alg);

PEPSKit.@autoopt @tensor env_init_peps.edges[4, 1, 2][χ1 D1 D2; χ2] * env_init_peps.corners[1, 2, 2][χ2; χ3] *
                         env_init_peps.edges[1, 2, 1][χ3 D3 D4; χ4] * env_init_peps.corners[2, 2, 2][χ4; χ5] * env_init_peps.edges[2, 1, 2][χ5 D5 D6; χ6] *
                         env_init_peps.corners[3, 2, 2][χ6; χ7] * env_init_peps.edges[3, 2, 1][χ7 D7 D8; χ8] * env_init_peps.corners[4, 2, 2][χ8; χ1] *
                         Ψ[1, 1][1; D3 D5 D7 D1] * conj(Ψ[1, 1][1; D4 D6 D8 D2])

PEPSKit.@autoopt @tensor env.edges[4][χ1 D1; χ2] * env.corners[1][χ2; χ3] * env.edges[1][χ3 D2; χ4] * env.corners[2][χ4; χ5] * env.edges[2][χ5 D3; χ6] *
                         env.corners[3][χ6; χ7] * env.edges[3][χ7 D4; χ8] * env.corners[4][χ8; χ1] * Z[1, 1][D1 D4; D2 D3]

# function env_init_peps_pullback(denv)
#     dP1 = zeros(ComplexF64, size(P1))
#     dP2 = zeros(ComplexF64, size(P2))
#     dU1 = zeros(ComplexF64, size(U1))
#     dU2 = zeros(ComplexF64, size(U2))

#     denv_edges = denv.edges
#     @tensor dP1[-1 -2; -3] := denv_edges[1, 2, 1][-1 -2 -3; -4] * P2[1; -4]
#     @tensor dP2[-1; -2 -3] := denv_edges[4, 1, 2][-1 -2 -3; -4] * P2[-4; -2 -3]
#     @tensor dU1[-1 -2; -3] := denv_edges[1, 2, 1][-1 -2 -3; -4] * P2[1; -4]
#     @tensor dU2[-1 -2; -3] := denv_edges[4, 1, 2][-1 -2 -3; -4] * P2[-4; -2 -3]

#     return NoTangent(), dP1, dP2, dU1, dU2
# end



