using MPSKitModels
using PEPSKit
using TensorKit
using KrylovKit
using OptimKit

function Kitaev_heisenberg(lattice::InfiniteSquare; kwargs...)
    return Kitaev_heisenberg(ComplexF64, Trivial, lattice; kwargs...)    
end

function Kitaev_heisenberg(T::Type{<:Number},
    S::Type{<:Sector},
    lattice::InfiniteSquare; ϕ::Number, h::Number)
    x_neighbors_σz = filter(n -> ((n[2].I[2] > n[1].I[2]) && ((n[1].I[2]%2 == 1 && n[1].I[1]%2 == 1) || (n[1].I[2]%2 == 0 && n[1].I[1]%2 == 0))), PEPSKit.nearest_neighbours(lattice))
    y_neighbors_σx = filter(n -> ((n[2].I[1] > n[1].I[1]) && ((n[1].I[2]%2 == 1 && n[1].I[1]%2 == 1) || (n[1].I[2]%2 == 0 && n[1].I[1]%2 == 0))), PEPSKit.nearest_neighbours(lattice))
    y_neighbors_σy = filter(n -> ((n[2].I[1] > n[1].I[1]) && ((n[1].I[2]%2 == 1 && n[1].I[1]%2 == 0) || (n[1].I[2]%2 == 0 && n[1].I[1]%2 == 1))), PEPSKit.nearest_neighbours(lattice))

    field = S_x() + S_y() + S_z()   
    spaces = fill(domain(field)[1], (lattice.Nrows, lattice.Ncols))
    return PEPSKit.LocalOperator(
        spaces, 
        ((idx,) => h*field for idx in PEPSKit.vertices(lattice))...,
        (neighbour => (cos(ϕ) + 2*sin(ϕ))*S_zz() + cos(ϕ)*(S_yy() + S_xx()) for neighbour in x_neighbors_σz)...,
        (neighbour => (cos(ϕ) + 2*sin(ϕ))*S_xx() + cos(ϕ)*(S_yy() + S_zz()) for neighbour in y_neighbors_σx)...,
        (neighbour => (cos(ϕ) + 2*sin(ϕ))*S_yy() + cos(ϕ)*(S_zz() + S_xx()) for neighbour in y_neighbors_σy)...,
    )

end

H = Kitaev_heisenberg(InfiniteSquare(2,2); ϕ=0, h=0)

D = 4
χ = 12

A = TensorMap(randn, ComplexF64, ℂ^2 ← ℂ^D⊗ℂ^D⊗(ℂ^D)'⊗(ℂ^1)')
B = TensorMap(randn, ComplexF64, ℂ^2 ← ℂ^D⊗ℂ^1⊗(ℂ^D)'⊗(ℂ^D)')

Ψ = InfinitePEPS([A B; B A])

ctm_alg = CTMRG()
opt_alg = PEPSOptimize(;
    boundary_alg=ctm_alg,
    optimizer=LBFGS(4; gradtol=1e-3, verbosity=2),
    gradient_alg=LinSolver(; iterscheme=:diffgauge),
)

env_init = leading_boundary(CTMRGEnv(Ψ, ComplexSpace(χ)), Ψ, ctm_alg);

result = fixedpoint(Ψ, H, opt_alg, env_init)