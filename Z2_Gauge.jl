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
χ = 24 # environment bond dimension
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




H = Fradkin_Shenker(InfiniteSquare(2,2); Jx=1, Jz=0, hx=0, hz=0, pdim=2, vdim=4)


Ψ = gauge_inv_peps(P, D, symm)
ctm_alg = CTMRG()
opt_alg = PEPSOptimize(;
    boundary_alg=ctm_alg,
    optimizer=LBFGS(4; gradtol=1e-3, verbosity=2),
    gradient_alg=LinSolver(; iterscheme=:diffgauge),
)

env_init = leading_boundary(CTMRGEnv(Ψ, Z2Space(0 => χ/2, 1 => χ/2)), Ψ, ctm_alg);

result = fixedpoint(Ψ, H, opt_alg, env_init)


file = jldopen("Initial_Psi.jld2", "w")
file["Ψ"] = Ψ
close(file)