using Pkg
Pkg.develop(path="./OptimKit.jl/")
Pkg.instantiate()
using Profile

hx_vals = 0.38:-0.01:0.25
hz = 0.11  # or any desired range
χ = 16 # environment bond dimension
D = 4 # PEPS bond dimension
#println("Running with: hx=$hx, hz=$hz, χ=$χ, D=$D")
include("new_toolbox.jl")

P = 2 # PEPS physical dimension
p = P / 2
v = Int(D / 2)
symm = Z2Irrep

PA = Z2Space(0 => p, 1 => p)
V = Z2Space(0 => v, 1 => v)
A = TensorMap(randn, ComplexF64, PA ← V ⊗ V ⊗ V' ⊗ V');

Ψ = peps_Gauge_trivial(A);
Ψ[1, 1] = my_symmetrize(Ψ[1, 1]);
A = Ψ[1, 1];
ctm_alg = SimultaneousCTMRG(; tol=1e-9, maxiter=200, verbosity=2)
env_init = CTMRGEnv(Ψ, Z2Space(0 => χ));
env_init = new_leading_boundary(env_init, Ψ, ctm_alg);
env = env_init

opt_alg = PEPSOptimize(;
    boundary_alg=ctm_alg,
    optimizer_alg=LBFGS(8; gradtol=1e-4, maxiter=200, verbosity=4),
    gradient_alg=EigSolver(;
        solver_alg=Arnoldi(; tol=1e-7, maxiter=3, verbosity=3, krylovdim=200, eager=true), iterscheme=:fixed))

for hx in hx_vals
    println(">>> Starting PEPS optimization for hx = $hx, hz = $hz")

    # Define Hamiltonian
    H = Fradkin_Shenker(InfiniteSquare(2, 2); Jx=1, Jz=1, hx=hx, hz=hz, pdim=2, vdim=4)

    # Optimize PEPS
    (A, env), E, ∂E, numfg, convhistory = optimize(
        (A, env), opt_alg.optimizer_alg;
        retract=my_retract_trivial,
        inner=my_inner_trivial,
        (transport!)=(my_transport_trivial!),
        (scale!)=my_scale!,
        (add!)=my_add!,
        (finalize!)=OptimKit._finalize!
    ) do (A, envs)
        E, gs = withgradient(A) do A
            Ψ = peps_Gauge_trivial(A)
            envs′ = hook_pullback(
                new_leading_boundary,
                envs,
                Ψ,
                opt_alg.boundary_alg;
                alg_rrule=opt_alg.gradient_alg,
            )
            ignore_derivatives() do
                opt_alg.reuse_env && update!(envs, envs′)
            end
            return costfun(Ψ, envs′, H)
        end
        gs = my_symmetrize(gs)
        return E, gs
    end

    # Update PEPS for next round
    new_Ψ = peps_Gauge_trivial(A)

    # Save to file
    outname = "final_Psi_reverse_trivial_1e4_hx=$(round(hx, digits=3))_hz=$(round(hz, digits=3))_χ=$(χ)_D=$(D).jld2"
    file = jldopen(outname, "w")
    file["Ψ"] = new_Ψ
    file["env"] = env
    file["E"] = E
    file["convhistory"] = convhistory
    close(file)

    println("✔ Saved: $outname")
end