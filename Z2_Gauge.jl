include("toolbox.jl")

χ = 10 # environment bond dimension
D = 4 # PEPS bond dimension
P = 2 # PEPS physical dimension
symm = Z2Irrep

# initialize states


ψ = gauge_inv_peps(P, D, symm)
ctm_alg = CTMRG()
opt_alg = PEPSOptimize(;
    boundary_alg=ctm_alg,
    optimizer=LBFGS(4; gradtol=1e-3, verbosity=2),
    gradient_alg=LinSolver(; iterscheme=:diffgauge),
)

