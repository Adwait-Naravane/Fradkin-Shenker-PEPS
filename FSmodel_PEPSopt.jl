using Pkg
Pkg.develop(path="./OptimKit.jl/")
Pkg.instantiate()
using Profile
# Get command-line arguments: hx, hz, χ, D
if length(ARGS) < 4
    error("Usage: julia FSmodel_PEPSopt.jl <hx> <hz> <χ> <D>")
end

hx = parse(Float64, ARGS[1])
hz = parse(Float64, ARGS[2])
χ  = parse(Int, ARGS[3])
D  = parse(Int, ARGS[4])
# hx = 0.34
# hz = 0.34
# χ = 16 # environment bond dimension
# D = 4 # PEPS bond dimension
println("Running with: hx=$hx, hz=$hz, χ=$χ, D=$D")
include("new_toolbox.jl")


P = 2 # PEPS physical dimension
p = P / 2
v = Int(D / 2)
symm = Z2Irrep

H = Fradkin_Shenker(InfiniteSquare(2, 2); Jx=1, Jz=1, hx=hx, hz=hz, pdim=2, vdim=4);


# file = jldopen("final_Psi_trivial_hx=$(hx)_hz=$(hz)_χ=$(χ)_D=$(D).jld2", "r")
# Ψ = file["Ψ"]
# env = file["env"]
# E = file["E"]
# convhistory = file["convhistory"]
# close(file)



PA = Z2Space(0 => p, 1 => p)
V = Z2Space(0 => v, 1 => v)
A = TensorMap(randn, ComplexF64, PA ← V ⊗ V ⊗ V' ⊗ V');
# # Be = exp.(1im*diag(rand(Float64,v,v)));
# # Bo = exp.(1im*diag(rand(Float64,v,v)));
# Be = diag(rand(Float64, v, v));
# Bo = diag(rand(Float64, v, v));

#Ψ = peps_Gauge(A, Be, Bo);
Ψ = peps_Gauge_trivial(A);
Ψ[1, 1] = my_symmetrize(Ψ[1, 1]);
A = Ψ[1, 1];
ctm_alg = SequentialCTMRG(; tol=1e-9, verbosity=2)
env_init = CTMRGEnv(Ψ, Z2Space(0 => χ));
env_init = new_leading_boundary(env_init, Ψ, ctm_alg);
#env_init = env


opt_alg = PEPSOptimize(;
    boundary_alg=ctm_alg,
    optimizer_alg=LBFGS(8; gradtol=1e-4, maxiter = 200, verbosity=4, linesearch = BackTrackingLineSearch()),
    gradient_alg=LinSolver(; iterscheme=:diffgauge),
    
)


# Ψ = peps_Gauge_trivial(A);
# env_init = env 
# ctm_alg = SequentialCTMRG(;tol = 1e-9, maxiter = 50, trscheme = truncdim(36), verbosity = 4)

# env_init = new_leading_boundary(env_init, Ψ, ctm_alg);

(A, env), E, ∂E, numfg, convhistory = optimize(
    (A, env_init), opt_alg.optimizer_alg; retract=my_retract_trivial, inner=my_inner_trivial, (transport!)=(my_transport_trivial!), (scale!)=my_scale!, (add!)=my_add!, (finalize!)=OptimKit._finalize!
) do (A, envs)
    E, gs = withgradient(A) do A
        Ψ = peps_Gauge_trivial(A)
        envs´ = hook_pullback(
            new_leading_boundary,
            envs,
            Ψ,
            opt_alg.boundary_alg;
            alg_rrule=opt_alg.gradient_alg,
        )
        ignore_derivatives() do
            opt_alg.reuse_env && update!(envs, envs´)
        end
        return costfun(Ψ, envs´, H)
    end
    gs = my_symmetrize(gs)
    return E, gs
end

# using ProfileSVG
# Profile.clear()

# ProfileSVG.@profview optimize(
#     (A, env_init), opt_alg.optimizer_alg;
#     retract=my_retract_trivial,
#     inner=my_inner_trivial,
#     (transport!)=my_transport_trivial!,
#     (scale!)=my_scale!,
#     (add!)=my_add!,
#     (finalize!)=OptimKit._finalize!,
# ) do (A, envs)
#     E, gs = withgradient(A) do A
#         Ψ = peps_Gauge_trivial(A)
#         envs´ = hook_pullback(
#             new_leading_boundary,
#             envs,
#             Ψ,
#             opt_alg.boundary_alg;
#             alg_rrule=opt_alg.gradient_alg,
#         )
#         ignore_derivatives() do
#             opt_alg.reuse_env && update!(envs, envs´)
#         end
#         return costfun(Ψ, envs´, H)
#     end
#     gs = my_symmetrize(gs)
#     return E, gs
# end

# open("profile_output.txt", "w") do io
#     Profile.print(io; format=:flat, sortedby=:count)
# end

# (A, Be, Bo, env), E, ∂E, numfg, convhistory = optimize(
#         (A, Be, Bo, env_init), opt_alg.optimizer_alg; retract = my_retract, inner=my_inner, (transport!)=(my_transport!), scale! = my_scale!, add! = my_add!, finalize! = OptimKit._finalize!
#     ) do (A, Be, Bo, envs)
#         E, gs = withgradient(A, Be, Bo) do A, Be, Bo
#             Ψ = peps_Gauge(A, Be, Bo)
#             envs´  = hook_pullback(
#                 new_leading_boundary,
#                 envs,
#                 Ψ,
#                 opt_alg.boundary_alg;
#                 alg_rrule=opt_alg.gradient_alg,
#             )
#             ignore_derivatives() do
#                 opt_alg.reuse_env && update!(envs, envs´)
#             end
#             return costfun(Ψ, envs´ , H)
#         end
#         gs = my_symmetrize(gs)
#         return E, gs
#     end


#new_Ψ = peps_Gauge(A, Be, Bo);
new_Ψ = peps_Gauge_trivial(A);

file = jldopen("final_Psi_trivial_1e4_hx=$(hx)_hz=$(hz)_χ=$(χ)_D=$(D).jld2", "w")
file["Ψ"] = new_Ψ
file["env"] = env
file["E"] = E
file["convhistory"] = convhistory
close(file)