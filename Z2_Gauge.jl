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

χ = 10 # environment bond dimension
D = 4 # PEPS bond dimension
P = 2 # PEPS physical dimension
symm = Z2Irrep

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

    p = 1
    PB = Z2Space(0 => 2*p)
    PA = Z2Space(0 => p, 1 => p)
    PT = Z2Space(0 => p)

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

end

function plaq(lattice::InfiniteSquare)
    neighbors = Tuple{CartesianIndex,CartesianIndex, CartesianIndex, CartesianIndex}[]
    for idx in vertices(lattice)
        if isodd(idx[1]) && isodd(idx[2])
            push!(neighbors, (idx + CartesianIndex(0,1), idx + CartesianIndex(1,2), idx + CartesianIndex(2,1), idx + CartesianIndex(1,0)))
        end
    end
    return neighbors
end

function triplebond(lattice::InfiniteSquare)
    neighbors = Tuple{CartesianIndex, CartesianIndex, CartesianIndex}[]
    for idx in vertices(lattice)
        if isodd(idx[1]) && isodd(idx[2])
            push!(neighbors, (idx, idx + CartesianIndex(0,1), idx + CartesianIndex(0,2)))
            push!(neighbors, (idx, idx + CartesianIndex(1,0), idx + CartesianIndex(2,0)))
        end
    end
    return neighbors
end


ψ = gauge_inv_peps(P, D, symm)
ctm_alg = CTMRG()
opt_alg = PEPSOptimize(;
    boundary_alg=ctm_alg,
    optimizer=LBFGS(4; gradtol=1e-3, verbosity=2),
    gradient_alg=LinSolver(; iterscheme=:diffgauge),
)

