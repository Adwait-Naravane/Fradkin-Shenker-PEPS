include("new_toolbox.jl")

χ = 16 # environment bond dimension
D = 4 # PEPS bond dimension
P = 2 # PEPS physical dimension
p = P/2
v = Int(D / 2)
symm = Z2Irrep

H = Fradkin_Shenker(InfiniteSquare(2,2); Jx=1, Jz=1, hx=1, hz=0, pdim=2, vdim=4)

PA = Z2Space(0 => p, 1 => p)
V = Z2Space(0 => v, 1 => v)
A = TensorMap(randn, ComplexF64, PA ← V ⊗ V ⊗ V' ⊗ V')
Be = diag(rand(Float64,v,v))
Bo = diag(rand(Float64,v,v))

Ψ = peps_Gauge(A, Be, Bo)
ctm_alg = SequentialCTMRG(verbosity = 4)
env_init  = new_leading_boundary(Ψ, ctm_alg, χ);


opt_alg = PEPSOptimize(;
    boundary_alg=ctm_alg,
    optimizer_alg=LBFGS(4; gradtol=1e-3, verbosity=2),
    gradient_alg=LinSolver(; iterscheme=:diffgauge),
)






(A, Be, Bo, env), E, ∂E, numfg, convhistory = optimize(
        (A, Be, Bo, env_init), opt_alg.optimizer_alg; retract = my_retract, inner=my_inner, scale! = my_scale!, add! = my_add!, finalize! = OptimKit._finalize!
    ) do (A, Be, Bo, envs)
        E, gs = withgradient(A, Be, Bo) do A, Be, Bo
            Ψ = peps_Gauge(A, Be, Bo)
            envs´  = hook_pullback(
                new_leading_boundary,
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
