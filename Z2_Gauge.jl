include("toolbox.jl")

χ = 10 # environment bond dimension
D = 4 # PEPS bond dimension
P = 2 # PEPS physical dimension
symm = Z2Irrep

# initialize states


function Fradkin_Shenker(lattice::InfiniteSquare; kwargs...)
    return Fradkin_Shenker(ComplexF64, Trivial, lattice; kwargs...)    
end

function Fradkin_Shenker(T::Type{<:Number},
    S::Type{<:Sector},
    lattice::InfiniteSquare; Jx::Number, Jz::Number, hx::Number, hz::Number)

    

end

ψ = gauge_inv_peps(P, D, symm)
ctm_alg = CTMRG()
opt_alg = PEPSOptimize(;
    boundary_alg=ctm_alg,
    optimizer=LBFGS(4; gradtol=1e-3, verbosity=2),
    gradient_alg=LinSolver(; iterscheme=:diffgauge),
)

