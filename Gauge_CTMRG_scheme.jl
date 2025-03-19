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