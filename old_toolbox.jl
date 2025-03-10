using LinearAlgebra
using TensorKit
using MPSKit
using PEPSKit
using Base.Iterators
using OptimKit
using KrylovKit
using VectorInterface
using Zygote
using ChainRulesCore
using Dates
using JLD2
using PEPSKit: PEPSTensor, CTMRGEnv, NORTH, SOUTH, WEST, EAST, NORTHWEST, NORTHEAST, SOUTHEAST, SOUTHWEST, _prev, _next, GradMode

# reduced density matrices (CTMRG version): 
function one_site_rho(ψ::InfinitePEPS, env::PEPSKit.CTMRGEnv)
    r, c = 1, 1
    @tensor opt=true ρ[-1; -2] :=
        env.edges[NORTH, r, c][1 2 3; 4] *
        env.corners[NORTHEAST, r, c][4; 5] *
        env.edges[EAST, r, c][5 6 7; 8] *
        env.corners[SOUTHEAST, r, c][8; 9] *
        env.edges[SOUTH, r, c][9 10 11; 12] *
        env.corners[SOUTHWEST, r, c][12; 13] *
        env.edges[WEST, r, c][13 14 15; 16] *
        env.corners[NORTHWEST, r, c][16; 1] *
        ψ[r, c][-1; 2 6 10 14] *
        conj(ψ[r, c][-2; 3 7 11 15])

    return ρ
end

function two_width_site_rho(ψ::InfinitePEPS, env::PEPSKit.CTMRGEnv)
    r, c = 1, 1
    @tensor opt=true ρ[-1 -2; -3 -4] :=
        env.edges[NORTH, r, c][1 2 3; 4] *
        env.edges[NORTH, r, c][4 5 6; 7] *
        env.corners[NORTHEAST, r, c][7; 8] *
        env.edges[EAST, r, c][8 9 10; 11] *
        env.corners[SOUTHEAST, r, c][11; 12] *
        env.edges[SOUTH, r, c][12 13 14; 15] *
        env.edges[SOUTH, r, c][15 16 17; 18] *
        env.corners[SOUTHWEST, r, c][18; 19] *
        env.edges[WEST, r, c][19 20 21; 22] *
        env.corners[NORTHWEST, r, c][22; 1] *
        ψ[r, c][-1; 2 23 16 20] *
        ψ[r, c][-2; 5 9 13 23] *
        conj(ψ[r, c][-3; 3 24 17 21]) *
        conj(ψ[r, c][-4; 6 10 14 24])

    return ρ
end

function two_depth_site_rho(ψ::InfinitePEPS, env::PEPSKit.CTMRGEnv)
    r, c = 1, 1
    @tensor opt=true ρ[-1 -2; -3 -4] :=
        env.edges[NORTH, r, c][1 2 3; 4] *
        env.corners[NORTHEAST, r, c][4; 5] *
        env.edges[EAST, r, c][5 6 7; 8] *
        env.edges[EAST, r, c][8 9 10; 11] *
        env.corners[SOUTHEAST, r, c][11; 12] *
        env.edges[SOUTH, r, c][12 13 14; 15] *
        env.corners[SOUTHWEST, r, c][15; 16] *
        env.edges[WEST, r, c][16 17 18; 19] *
        env.edges[WEST, r, c][19 20 21; 22] *
        env.corners[NORTHWEST, r, c][22; 1] *
        ψ[r, c][-1; 2 6 23 20] *
        ψ[r, c][-2; 23 9 17 13] *
        conj(ψ[r, c][-3; 3 7 24 21])*
        conj(ψ[r, c][-4; 24 10 18 14])

    return ρ
end

function four_site_rho(ψ::InfinitePEPS, env::PEPSKit.CTMRGEnv)
    r, c = 1, 1
    @tensor opt=true ρ[-1 -2 -3 -4; -5 -6 -7 -8] :=
        env.edges[NORTH, r, c][1 2 3; 4] *
        env.edges[NORTH, r, c][4 5 6; 7] *
        env.corners[NORTHEAST, r, c][7; 8] *
        env.edges[EAST, r, c][8 9 10; 11] *
        env.edges[EAST, r, c][11 12 13; 14] *
        env.corners[SOUTHEAST, r, c][14; 15] *
        env.edges[SOUTH, r, c][15 16 17; 18] *
        env.edges[SOUTH, r, c][18 19 20; 21] *
        env.corners[SOUTHWEST, r, c][21; 22] *
        env.edges[WEST, r, c][22 23 24; 25] *
        env.edges[WEST, r, c][25 26 27; 28] *
        env.corners[NORTHWEST, r, c][28; 1] *
        ψ[r, c][-1; 2 30 31 26] *
        ψ[r, c][-2; 5 9 32 30] *
        ψ[r, c][-3; 31 33 19 23] *
        ψ[r, c][-4; 32 12 16 33] *
        conj(ψ[r, c][-5; 3 34 35 27])*
        conj(ψ[r, c][-6; 6 10 36 34])*
        conj(ψ[r, c][-7; 35 37 20 24])*
        conj(ψ[r, c][-8; 36 13 17 37])

    return ρ
end


# Normalized expectation values for blocked PEPS:
function onesite_expval_gauge(ψ::InfinitePEPS, env::CTMRGEnv, O::AbstractTensorMap{S,1,1}, splitter::TensorMap) where {S}
    ρ = one_site_rho(ψ, env)
    @tensor nn = ρ[1; 1]
    @tensor opt=true Eh = ρ[2; 1] * splitter[1; 3 4 5 6] * O[4; 7] * conj(splitter[2; 3 7 5 6]) + ρ[2; 1] * splitter[1; 3 4 5 6] * O[5; 8] * conj(splitter[2; 3 4 8 6])
    return Eh / nn
end

function onesite_expval_matter(ψ::InfinitePEPS, env::CTMRGEnv, O::AbstractTensorMap{S,1,1}, splitter::TensorMap) where {S}
    ρ = one_site_rho(ψ, env)
    @tensor nn = ρ[1; 1]
    @tensor opt=true Eh = ρ[2; 1] * splitter[1; 3 4 5 6] * O[3; 7] * conj(splitter[2; 7 4 5 6])
    return Eh / nn
end

function bonds_expval(ψ::InfinitePEPS, env::CTMRGEnv, O::AbstractTensorMap{S,3,3}, splitter::TensorMap) where {S}
    ρ1 = two_width_site_rho(ψ, env)
    ρ2 = two_depth_site_rho(ψ, env)
    @tensor opt=true nn1 = ρ1[1 2; 1 2]
    @tensor opt=true nn2 = ρ2[1 2; 1 2]
    @tensor opt=true Eh1 = ρ1[1 2; 3 4] * conj(splitter[1; 5 6 7 8]) * conj(splitter[2; 9 10 11 12]) * O[13 14 15; 5 7 9] * splitter[3; 13 6 14 8] * splitter[4; 15 10 11 12]
    @tensor opt=true Eh2 = ρ2[1 2; 3 4] * conj(splitter[1; 5 6 7 8]) * conj(splitter[2; 9 10 11 12]) * O[13 14 15; 5 6 9] * splitter[3; 13 14 7 8] * splitter[4; 15 10 11 12]
    return Eh1 / nn1 + Eh2 / nn2
end

function plaq_expval(ψ::InfinitePEPS, env::CTMRGEnv, O::AbstractTensorMap{S,4,4}, splitter::TensorMap) where {S}
    ρ = four_site_rho(ψ, env)
    @tensor opt=true nn = ρ[1 2 3 4; 1 2 3 4]
    @tensor opt=true Eh = ρ[1 2 3 4; 5 6 7 8] * conj(splitter[1; 9 10 11 12]) * conj(splitter[2; 13 14 15 16]) * conj(splitter[3; 17 18 19 20]) * conj(splitter[4; 21 22 23 24]) * splitter[5; 9 25 26 12] * splitter[6; 13 27 15 16] * splitter[7; 17 18 28 20] * splitter[8; 21 22 23 24] * O[25 26 27 28; 10 11 14 19]
    return Eh / nn
end

function gauge_symm(ψ::InfinitePEPS, env::CTMRGEnv, GZ::AbstractTensorMap{S,1,1}, Z::AbstractTensorMap{S,1,1}, splitter::TensorMap) where {S}
    ρ = four_site_rho(ψ, env)
    @tensor opt=true nn = ρ[1 2 3 4; 1 2 3 4]
    @tensor opt=true Eh = ρ[1 2 3 4; 5 6 7 8] * conj(splitter[1; 9 10 11 12]) * conj(splitter[2; 13 14 15 16]) * conj(splitter[3; 17 18 19 20]) * conj(splitter[4; 21 22 23 24]) * splitter[5; 9 10 11 12] * splitter[6; 13 25 15 16] * splitter[7; 17 18 26 20] * splitter[8; 27 28 29 24] * GZ[25;14]* GZ[26;19]* Z[27;21]* GZ[28;22]* GZ[29;23]
    return real(Eh / nn)
end

function tHooft_string(ψ::InfinitePEPS, env::CTMRGEnv, GZ::AbstractTensorMap{S,1,1}, splitter::TensorMap, NGZ::Int64) where {S}
    r, c = 1, 1
    global n = 0

    @tensor opt=true left[-1 -2 -3; -4] :=
        env.corners[SOUTHWEST, r, c][-1; 1] *
        env.edges[WEST, r, c][1 -2 -3; 4] *
        env.corners[NORTHWEST, r, c][4; -4]

    @tensor opt=true leftnorm[-1 -2 -3; -4] :=
        env.corners[SOUTHWEST, r, c][-1; 1] *
        env.edges[WEST, r, c][1 -2 -3; 4] *
        env.corners[NORTHWEST, r, c][4; -4]

    while n < NGZ
        @tensor left[-1 -2 -3; -4] =
            left[1 2 3; 4] *
            env.edges[NORTH, r, c][4 5 6; -4] *
            env.edges[SOUTH, r, c][-1 9 10; 1] *
            ψ[r, c][11; 5 -2 9 2] *
            conj(splitter[11; 12 13 14 15]) *
            GZ[16; 13] * 
            splitter[17; 12 16 14 15] *
            conj(ψ[r, c][17; 6 -3 10 3])
        
        @tensor leftnorm[-1 -2 -3; -4] =
            leftnorm[1 2 3; 4] *
            env.edges[NORTH, r, c][4 5 6; -4] *
            env.edges[SOUTH, r, c][-1 9 10; 1] *
            ψ[r, c][11; 5 -2 9 2] *
            conj(ψ[r, c][11; 6 -3 10 3])

        global n += 1

    end
    
    @tensor finitestring = left[1 2 3; 4] *
        env.corners[NORTHEAST, r, c][4; 5] *
        env.edges[EAST, r, c][5 2 3; 6] *
        env.corners[SOUTHEAST, r, c][6; 1]
    
    @tensor leftnorm = leftnorm[1 2 3; 4] *
        env.corners[NORTHEAST, r, c][4; 5] *
        env.edges[EAST, r, c][5 2 3; 6] *
        env.corners[SOUTHEAST, r, c][6; 1]

    return real(finitestring/leftnorm)

end

function Wilson_string(ψ::InfinitePEPS, env::CTMRGEnv, GX::AbstractTensorMap{S,1,1}, XX::AbstractTensorMap{S,2,2}, splitter::TensorMap, NGZ::Int64) where {S}
    r, c = 1, 1
    global n = 1

    @tensor opt=true left[-1 -2 -3; -4] :=
        env.corners[SOUTHWEST, r, c][-1; 1] *
        env.edges[WEST, r, c][1 -2 -3; 4] *
        env.corners[NORTHWEST, r, c][4; -4]

    @tensor opt=true leftnorm[-1 -2 -3; -4] :=
        env.corners[SOUTHWEST, r, c][-1; 1] *
        env.edges[WEST, r, c][1 -2 -3; 4] *
        env.corners[NORTHWEST, r, c][4; -4]
    
    @tensor left[-1 -2 -3 -5; -4 -6] :=
        left[1 2 3; 4] *
        env.edges[NORTH, r, c][4 5 6; -4] *
        env.edges[SOUTH, r, c][-1 9 10; 1] *
        ψ[r, c][11; 5 -2 9 2] *
        conj(splitter[11; 12 13 14 15]) *
        GX[16; 14] * 
        XX[18 -5; 12 -6] *
        splitter[17; 18 13 16 15] *
        conj(ψ[r, c][17; 6 -3 10 3])
    
    @tensor leftnorm[-1 -2 -3; -4] =
        leftnorm[1 2 3; 4] *
        env.edges[NORTH, r, c][4 5 6; -4] *
        env.edges[SOUTH, r, c][-1 9 10; 1] *
        ψ[r, c][11; 5 -2 9 2] *
        conj(ψ[r, c][11; 6 -3 10 3])

    while n < NGZ
        @tensor left[-1 -2 -3 -5; -4 -6] =
            left[1 2 3 -5; 4 -6] *
            env.edges[NORTH, r, c][4 5 6; -4] *
            env.edges[SOUTH, r, c][-1 9 10; 1] *
            ψ[r, c][11; 5 -2 9 2] *
            conj(splitter[11; 12 13 14 15]) *
            GX[16; 14] * 
            splitter[17; 12 13 16 15] *
            conj(ψ[r, c][17; 6 -3 10 3])
    
        @tensor leftnorm[-1 -2 -3; -4] =
            leftnorm[1 2 3; 4] *
            env.edges[NORTH, r, c][4 5 6; -4] *
            env.edges[SOUTH, r, c][-1 9 10; 1] *
            ψ[r, c][11; 5 -2 9 2] *
            conj(ψ[r, c][11; 6 -3 10 3])

        global n += 1

    end

    @tensor left[-1 -2 -3; -4] :=
        left[1 2 3 30; 4 31] *
        env.edges[NORTH, r, c][4 5 6; -4] *
        env.edges[SOUTH, r, c][-1 9 10; 1] *
        ψ[r, c][11; 5 -2 9 2] *
        conj(splitter[11; 31 13 14 15]) *
        splitter[17; 30 13 14 15] *
        conj(ψ[r, c][17; 6 -3 10 3])

    @tensor leftnorm[-1 -2 -3; -4] =
        leftnorm[1 2 3; 4] *
        env.edges[NORTH, r, c][4 5 6; -4] *
        env.edges[SOUTH, r, c][-1 9 10; 1] *
        ψ[r, c][11; 5 -2 9 2] *
        conj(ψ[r, c][11; 6 -3 10 3])
    
    @tensor finitestring = left[1 2 3; 4] *
        env.corners[NORTHEAST, r, c][4; 5] *
        env.edges[EAST, r, c][5 2 3; 6] *
        env.corners[SOUTHEAST, r, c][6; 1]
    
    @tensor leftnorm = leftnorm[1 2 3; 4] *
        env.corners[NORTHEAST, r, c][4; 5] *
        env.edges[EAST, r, c][5 2 3; 6] *
        env.corners[SOUTHEAST, r, c][6; 1]

    return real(finitestring/leftnorm)

end

function random_peps end
function random_peps(pdim::Int, vdim::Int, ::Type{Trivial}; unitcell=(2, 2))
    P = ℂ^pdim
    V = ℂ^vdim
    return InfinitePEPS(P, V; unitcell)
end
function random_peps(pdim::Int, vdim::Int, ::Type{Z2Irrep}; unitcell=(2, 2))
    P = Z2Space(0 => pdim/2, 1 => pdim/2)
    V = Z2Space(0 => vdim/2, 1 => vdim/2)
    return InfinitePEPS(P, V; unitcell)
end


"""
Make a gauge-invariant PEPS.
"""
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

"""
Naive symmetrize PEPS unit cell in depth direction.
"""
naive_symmetrize!(ψ::InfinitePEPS) = PEPSKit.symmetrize(ψ, PEPSKit.Full())

"""
Fully symmetrize first site of PEPS unit cell.
"""
function first_site_symmetrize!(ψ::InfinitePEPS)
    ψ[1, 1] = PEPSKit.herm_depth_inv(PEPSKit.rot_inv(ψ[1, 1]))
    return ψ
end

"""
    random_env(ψ::InfinitePEPS, vdim::Int, symmetry::Type{<:Sector})

Initialize a random CTMRG environment with a given bond dimension.
"""
function random_env end
function random_env(ψ::InfinitePEPS, vdim::Int, ::Type{Trivial})
    Venv = ℂ^vdim
    return CTMRGEnv(ψ; Venv)
end
function random_env(ψ::InfinitePEPS, vdim::Int, ::Type{Z2Irrep})
    Venv = Z2Space(0 => vdim/2, 1 => vdim/2)
    return CTMRGEnv(ψ; Venv)
end

"""
    Z_op(symmetry::Type{<:Sector})

Returns a single-site Z operator for a given symmetry.
"""
function Z_op end
function Z_op(::Type{Trivial})
    P = ℂ^2
    return TensorMap(ComplexF64[1 0; 0 -1], P ← P)
end
function Z_op(::Type{Z2Irrep})
    P = Z2Space(0 => 1, 1 => 1)
    Z = TensorMap(zeros, ComplexF64, P ← P)
    block(Z, Z2Irrep(0)) .= 1
    block(Z, Z2Irrep(1)) .= -1
    return Z
end

# Blocking and splitting tools
# ----------------------------

# space getters
north_spaces(ψ::InfinitePEPS) = map(c -> space(ψ[1, c], NORTH + 1), 1:size(ψ, 2))
south_spaces(ψ::InfinitePEPS) = map(c -> space(ψ[end, c], SOUTH + 1), 1:size(ψ, 2))
east_spaces(ψ::InfinitePEPS) = map(r -> space(ψ[r, end], EAST + 1), 1:size(ψ, 1))
west_spaces(ψ::InfinitePEPS) = map(r -> space(ψ[r, 1], WEST + 1), 1:size(ψ, 1))
phys_spaces(ψ::InfinitePEPS) = reshape(map(((r, c),) -> space(ψ[r, c], 1), Iterators.product(axes(ψ)...)), length(ψ))

# PEPS blocking routine specifically for 2 x 2 unit cell
function block_peps(ψ::InfinitePEPS)

    # getting fusers is generic
    nsp = conj.(north_spaces(ψ))
    f_north = isometry(storagetype(ψ[1, 1]), prod(nsp) ← fuse(nsp...))
    esp = conj.(east_spaces(ψ))
    f_east = isometry(storagetype(ψ[1, end]), prod(esp) ← fuse(esp...))
    ssp = south_spaces(ψ)
    f_south = isometry(storagetype(ψ[end, 1]), fuse(ssp...) ← prod(ssp))
    wsp = west_spaces(ψ)
    f_west = isometry(storagetype(ψ[1, 1]), fuse(wsp...) ← prod(wsp))
    psp = phys_spaces(ψ)
    f_phys = isometry(storagetype(ψ[1, 1]), fuse(psp...) ← prod(psp))
        
    ψ_blocked = block_peps(ψ, f_north, f_east, f_south, f_west, f_phys)

    return ψ_blocked, f_north, f_east, f_south, f_west, f_phys
end

function block_peps(ψ::InfinitePEPS, f_north, f_east, f_south, f_west, f_phys)

    # contraction is specific for now, to be generalized
    @tensor opt=true Ablocked[-1; -2 -3 -4 -5] :=
        f_phys[-1; 13 14 15 16] *
        f_north[2 3; -2] *
        f_east[5 6; -3] *
        f_south[-4; 8 9] *
        f_west[-5; 11 12] *
        ψ[1, 1][13; 2 1 10 11] *
        ψ[2, 1][14; 10 7 8 12] *
        ψ[1, 2][15; 3 5 4 1] *
        ψ[2, 2][16; 4 6 9 7]
        
    return InfinitePEPS(Ablocked)
end


# Making the V2B map:
function V2B(ψ_blocked::InfinitePEPS, pdim::Int, vdim::Int)
    P = Int(pdim)
    D = Int(vdim)

    Vec_R = [0.0]
    Vec_I = [0.0]
    Ind_R = zeros(Int, (4*P, D, D, D, D))
    Ind_I = zeros(Int, (4*P, D, D, D, D))
    M = convert(Array, ψ_blocked[1])

    for p in 1:4*P, n in 1:D, o in 1:D, z in 1:D, w in 1:D
        rounded_val_R = round(real(M[p, n, o, z, w]), digits=14)
        rounded_val_I = round(imag(M[p, n, o, z, w]), digits=14)
        if rounded_val_R in Vec_R
            Ind_R[p, n, o, z, w] = findfirst(x -> x == rounded_val_R, Vec_R)
        else
            push!(Vec_R, rounded_val_R)
            Ind_R[p, n, o, z, w] = findfirst(x -> x == rounded_val_R, Vec_R)
        end
        if rounded_val_I in Vec_I
            Ind_I[p, n, o, z, w] = findfirst(x -> x == rounded_val_I, Vec_I)
        else
            push!(Vec_I, rounded_val_I)
            Ind_I[p, n, o, z, w] = findfirst(x -> x == rounded_val_I, Vec_I)
        end
    end

    #TODO: shorten this
    Vec_R = Vec_R[2:end]
    Vec_I = Vec_I[2:end]
    Vec = Vector{Vector{Float64}}()
    push!(Vec, Vec_R)
    push!(Vec, Vec_I)
    Ind = Vector{Array{Int64, 5}}()
    push!(Ind, Ind_R)
    push!(Ind, Ind_I)
    numvar = size(Vec_R)[1] + size(Vec_I)[1]
    println("\nOptimization on $numvar floats:")

    return Vec, Ind
end 

function Z2GaugeHam(pdim::Int, vdim::Int, ::Type{Z2Irrep})

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

    return Z, GZ, GX, XX, PLAQ, B
end

# TODO: make sure Zygote knows how to derive this one
function H_expectation_value(
    ψ::InfinitePEPS, env::PEPSKit.CTMRGEnv, H::Tuple, params::NTuple{4,Float64}, splitter::TensorMap
)
    J_x, J_z, h_x, h_z = params
    Z, GZ, GX, XX, PLAQ, B = H

    E = 0.0

    E -= J_x * onesite_expval_matter(ψ, env, Z, splitter)
    E -= J_z * plaq_expval(ψ, env, PLAQ, splitter)
    E -= h_x * onesite_expval_gauge(ψ, env, GZ, splitter)
    E -= h_z * bonds_expval(ψ, env, B, splitter)

    return real(E)
end

# Post processing
# ---------------

_conj(t::AbstractTensorMap{S, N, 1}) where {S, N} = permute(t', ((Tuple(2:N+1)), (1,)))

function A_from_jld(filename::AbstractString)
    my_array = nothing
    jldopen(filename, "r") do file
        my_array = file["my_array"]
    end
    return my_array
end

# struct my_dict
#     H::Dict
#     peps::InfinitePEPS
#     env::CTMRGEnv
#     fusers::
# end

function read_from_jld(filename::AbstractString)
    saved_H, saved_peps, saved_env, saved_fusers = nothing, nothing, nothing, nothing
    jldopen(filename, "r") do file
        saved_H = file["H"]
        saved_peps = file["peps"]
        saved_env = file["env"]
        saved_fusers = file["fusers"]
    end

    return saved_H, saved_peps, saved_env, saved_fusers
end

function write_arrays_to_file(filename::AbstractString, arrays::Vector{Vector{T}}) where T
    open(filename, "w") do file
        for array in arrays
            println(file, join(array, ", "))
        end
    end
end

function Print_expectation_values(ψ_blocked::InfinitePEPS, env_blocked::PEPSKit.CTMRGEnv, H::Tuple, f_phys::TensorMap)
    Z, GZ, GX, XX, PLAQ, B = H

    muz = onesite_expval_matter(ψ_blocked, env_blocked, Z, f_phys)
    sigz = onesite_expval_gauge(ψ_blocked, env_blocked, GZ, f_phys)/2
    bonds = bonds_expval(ψ_blocked, env_blocked, B, f_phys)/2
    plaq = plaq_expval(ψ_blocked, env_blocked, PLAQ, f_phys)

    println("\n<ψ|μz|ψ>/<ψ|ψ> = ", muz)
    println("<ψ|σz|ψ>/<ψ|ψ> = ", sigz)
    println("<ψ|bond|ψ>/<ψ|ψ> = ", bonds)
    println("<ψ|plaq|ψ>/<ψ|ψ> = ", plaq, "\n")

    return muz, sigz, bonds, plaq
end

function corner_entropy(env::PEPSKit.CTMRGEnv)
    C1 = env.corners[1, 1, 1]
    C2 = env.corners[2, 1, 1]
    C3 = env.corners[3, 1, 1]
    C4 = env.corners[4, 1, 1]

    χ = Int(dim(domain(C1))/1)

    @tensor C[-1; -2] := C1[-1; 1] * C2[1; 2] * C3[2; 3] * C4[3; -2]
    E, V = eig(C)

    Corner_entropy = 0.0
    for (s, f) in fusiontrees(E)
        for d in 1:χ
            Corner_entropy -= E[s, f][d, d] * log(E[s, f][d, d])
        end
    end

    return Corner_entropy
end

function correlation_length_boundaryMPS(ψ_blocked, alg_VUMPS::VUMPS, χ::Int64, sector)

    V_MPS = Z2Space(0 => Int(χ/2), 1 => Int(χ/2))

    trans = PEPSKit.InfiniteTransferPEPS(ψ_blocked, 1, 1)
    mps = PEPSKit.initializeMPS(trans, [V_MPS])
    mps, _ = leading_boundary(mps, trans, alg_VUMPS)

    return correlation_length(mps; sector=sector) 
end

function strings_VUMPS(ψ_blocked::InfinitePEPS, alg_VUMPS::VUMPS, f_phys::TensorMap, χ::Int64, GX::AbstractTensorMap{S,1,1}, GZ::AbstractTensorMap{S,1,1}) where {S}

    D = Int(((dim(domain(ψ_blocked[1])))^(1/4))/2)
    V = Z2Space(0 => D, 1 => D)
    V_MPS = Z2Space(0 => Int(χ/2), 1 => Int(χ/2))

    trans_t = PEPSKit.InfiniteTransferPEPS(ψ_blocked, 1, 1)
    trans_b = dagger(trans_t)

    mps_t = PEPSKit.initializeMPS(trans_t, [V_MPS])
    mps_b = PEPSKit.initializeMPS(trans_b, [V_MPS])

    env_VUMPS_t = leading_boundary(mps_t, trans_t, alg_VUMPS)
    env_VUMPS_b = leading_boundary(mps_b, trans_b, alg_VUMPS)

    TOP = env_VUMPS_t[1].AL[1]
    BOT = env_VUMPS_b[1].AL[1]

    vals_Wilson, vecs, info =
        eigsolve(TensorMap(randn, scalartype(TOP), space(BOT, 1) ⊗ V' ⊗ V ← space(TOP, 1)), 1, :LM) do v

            @tensor opt=true vout[-4 -1 -2; -3] := 
            TOP[1 8 9; -3] *
            conj(BOT[2 3 6; -4]) *
            ψ_blocked[1, 1][5; 8 -1 3 4] *
            f_phys[15; 10 11 12 13] * 
            GX[11; 14] * 
            conj(f_phys[5; 10 14 12 13]) *
            conj(ψ_blocked[1, 1][15; 9 -2 6 7]) *
            v[2 4 7; 1]

        end

    vals_Wilson_odd, vecs, info =
        eigsolve(TensorMap(randn, scalartype(TOP), space(BOT, 1) ⊗ V' ⊗ V ⊗ Z2Space(1 => 1) ← space(TOP, 1)), 1, :LM) do v

            @tensor opt=true vout[-4 -1 -2 -5; -3] := 
            TOP[1 8 9; -3] *
            conj(BOT[2 3 6; -4]) *
            ψ_blocked[1, 1][5; 8 -1 3 4] *
            f_phys[15; 10 11 12 13] * 
            GX[11; 14] * 
            conj(f_phys[5; 10 14 12 13]) *
            conj(ψ_blocked[1, 1][15; 9 -2 6 7]) *
            v[2 4 7 -5; 1]

        end

    vals_tHooft, vecs, info =
        eigsolve(TensorMap(randn, scalartype(TOP), space(BOT, 1) ⊗ V' ⊗ V ← space(TOP, 1)), 1, :LM) do v

            @tensor opt=true vout[-4 -1 -2; -3] := 
            TOP[1 8 9; -3] *
            conj(BOT[2 3 6; -4]) *
            ψ_blocked[1, 1][5; 8 -1 3 4] *
            f_phys[15; 10 11 12 13] * 
            GZ[11; 14] * 
            conj(f_phys[5; 10 14 12 13]) *
            conj(ψ_blocked[1, 1][15; 9 -2 6 7]) *
            v[2 4 7; 1]

        end
    
    vals_tHooft_odd, vecs, info =
        eigsolve(TensorMap(randn, scalartype(TOP), space(BOT, 1) ⊗ V' ⊗ V ⊗ Z2Space(1 => 1) ← space(TOP, 1)), 1, :LM) do v

            @tensor opt=true vout[-4 -1 -2 -5; -3] := 
            TOP[1 8 9; -3] *
            conj(BOT[2 3 6; -4]) *
            ψ_blocked[1, 1][5; 8 -1 3 4] *
            f_phys[15; 10 11 12 13] * 
            GZ[11; 14] * 
            conj(f_phys[5; 10 14 12 13]) *
            conj(ψ_blocked[1, 1][15; 9 -2 6 7]) *
            v[2 4 7 -5; 1]

        end

    return vals_Wilson, vals_Wilson_odd, vals_tHooft, vals_tHooft_odd
end

function strings_CTMRG(ψ_blocked::InfinitePEPS, env::PEPSKit.CTMRGEnv, f_phys::TensorMap, GX::AbstractTensorMap{SS,1,1}, GZ::AbstractTensorMap{SS,1,1}) where {SS}

    D = Int(((dim(domain(ψ_blocked[1])))^(1/4))/2)
    V = Z2Space(0 => D, 1 => D)

    N = env.edges[NORTH, 1, 1]
    S = env.edges[SOUTH, 1, 1]

    vals_triv, vecs, info =
        eigsolve(TensorMap(randn, scalartype(N), space(N, 1) ⊗ V' ⊗ V ← space(N, 1)), 1, :LM) do v
            
            @tensor opt=true vout[-4 -1 -2; -3] := 
            N[1 8 9; -3] *
            S[-4 3 6; 2] *
            ψ_blocked[1, 1][5; 8 -1 3 4] *
            f_phys[15; 10 11 12 13] * 
            conj(f_phys[5; 10 11 12 13]) *
            conj(ψ_blocked[1, 1][15; 9 -2 6 7]) *
            v[2 4 7; 1]

        end
    
    norm_odd, vecs_wilson, info =
        eigsolve(TensorMap(randn, scalartype(N), space(N, 1) ⊗ V' ⊗ V ⊗ Z2Space(1 => 1) ← space(N, 1)), 1, :LM) do v
                
            @tensor opt=true vout[-4 -1 -2 -5; -3] := 
            N[1 8 9; -3] *
            S[-4 3 6; 2] *
            ψ_blocked[1, 1][5; 8 -1 3 4] *
            f_phys[15; 10 11 12 13] * 
            conj(f_phys[5; 10 11 12 13]) *
            conj(ψ_blocked[1, 1][15; 9 -2 6 7]) *
            v[2 4 7 -5; 1]

        end
    
    vals_Wilson_right, vecs_wilson_right, info =
        eigsolve(TensorMap(randn, scalartype(N), space(N, 1) ← space(N, 1) ⊗ V' ⊗ V ⊗ Z2Space(1 => 1)), 1, :LM) do v
                
            @tensor opt=true vout[-3; -4 -1 -2 -5] := 
            N[-3 8 9; 1] *
            S[2 3 6; -4] *
            ψ_blocked[1, 1][5; 8 4 3 -1] *
            f_phys[15; 10 11 12 13] * 
            GX[12; 14] * 
            conj(f_phys[5; 10 11 14 13]) *
            conj(ψ_blocked[1, 1][15; 9 7 6 -2]) *
            v[1; 2 4 7 -5]

        end

    vals_Wilson_left, vecs_wilson_left, info =
        eigsolve(TensorMap(randn, scalartype(N), space(N, 1) ⊗ V' ⊗ V ⊗ Z2Space(1 => 1) ← space(N, 1)), 1, :LM) do v
                
            @tensor opt=true vout[-4 -1 -2 -5; -3] := 
            N[1 8 9; -3] *
            S[-4 3 6; 2] *
            ψ_blocked[1, 1][5; 8 -1 3 4] *
            f_phys[15; 10 11 12 13] * 
            GX[12; 14] * 
            conj(f_phys[5; 10 11 14 13]) *
            conj(ψ_blocked[1, 1][15; 9 -2 6 7]) *
            v[2 4 7 -5; 1]

        end

    vals_tHooft, vecs, info =
        eigsolve(TensorMap(randn, scalartype(N), space(N, 1) ⊗ V' ⊗ V ← space(N, 1)), 1, :LM) do v
                
            @tensor opt=true vout[-4 -1 -2; -3] := 
            N[1 8 9; -3] *
            S[-4 3 6; 2] *
            ψ_blocked[1, 1][5; 8 -1 3 4] *
            f_phys[15; 10 11 12 13] * 
            GZ[11; 14] * 
            conj(f_phys[5; 10 14 12 13]) *
            conj(ψ_blocked[1, 1][15; 9 -2 6 7]) *
            v[2 4 7; 1]

        end
    
    vals_tHooft_odd, vecs, info =
        eigsolve(TensorMap(randn, scalartype(N), space(N, 1) ⊗ V' ⊗ V ⊗ Z2Space(1 => 1) ← space(N, 1)), 1, :LM) do v
                
            @tensor opt=true vout[-4 -1 -2 -5; -3] := 
            N[1 8 9; -3] *
            S[-4 3 6; 2] *
            ψ_blocked[1, 1][5; 8 -1 3 4] *
            f_phys[15; 10 11 12 13] * 
            GZ[11; 14] * 
            conj(f_phys[5; 10 14 12 13]) *
            conj(ψ_blocked[1, 1][15; 9 -2 6 7]) *
            v[2 4 7 -5; 1]

        end
    
        # @tensor opt=true Wilsoncontracted[-1; -2] := vecs_wilson_left[1][4 1 2 -1; 3] * vecs_wilson_right[1][3; 4 1 2 -2]
        # @tensor opt=true WFC = Wilsoncontracted[1; 1]
        # println(abs(WFC))
        # println(vals_Wilson_left[1])
        # println(vals_Wilson_right[1])

    return vals_triv, norm_odd, vals_Wilson_left, vals_Wilson_right, vals_tHooft, vals_tHooft_odd
end


function shadows_CTMRG(ψ_blocked::InfinitePEPS, env::PEPSKit.CTMRGEnv, f_north::TensorMap)

    D = Int(((dim(domain(ψ_blocked[1])))^(1/4))/2)
    V = Z2Space(0 => D, 1 => D)

    Z_virt = TensorMap(zeros, ComplexF64, V ← V)

    for (s, f) in fusiontrees(Z_virt)
        if s.uncoupled[1] == Irrep[ℤ₂](0)
            Z_virt[s, f] = Matrix{ComplexF64}(I, D, D)
        else
            Z_virt[s, f] = Matrix{ComplexF64}(-I, D, D)
        end
    end

    N = env.edges[NORTH, 1, 1]
    Z = env.edges[SOUTH, 1, 1]
    
    II, vecs, info =
        eigsolve(TensorMap(randn, scalartype(N), space(N, 1) ⊗ V' ⊗ V ← space(N, 1)), 1, :LM) do v
            
            @tensor opt=true vout[-4 -1 -2; -3] := 
            N[1 10 9; -3] *
            Z[-4 3 6; 2] *
            f_north[11 12; 10] *
            conj(f_north[11 12; 13]) *
            ψ_blocked[1, 1][5; 13 -1 3 4] *
            conj(ψ_blocked[1, 1][5; 9 -2 6 7]) *
            v[2 4 7; 1]

        end
    
    ZI, vecs, info =
        eigsolve(TensorMap(randn, scalartype(N), space(N, 1) ⊗ V' ⊗ V ← space(N, 1)), 1, :LM) do v
            
            @tensor opt=true vout[-4 -1 -2; -3] := 
            N[1 10 9; -3] *
            Z[-4 3 6; 2] *
            f_north[11 12; 10] *
            Z_virt[14; 11] *
            conj(f_north[14 12; 13]) *
            ψ_blocked[1, 1][5; 13 -1 3 4] *
            conj(ψ_blocked[1, 1][5; 9 -2 6 7]) *
            v[2 4 7; 1]

        end
    
    IZ, vecs, info =
        eigsolve(TensorMap(randn, scalartype(N), space(N, 1) ⊗ V' ⊗ V ← space(N, 1)), 1, :LM) do v
            
            @tensor opt=true vout[-4 -1 -2; -3] := 
            N[1 8 10; -3] *
            Z[-4 3 6; 2] *
            conj(f_north[11 12; 10]) *
            Z_virt[11; 14] *
            f_north[14 12; 13] *
            ψ_blocked[1, 1][5; 8 -1 3 4] *
            conj(ψ_blocked[1, 1][5; 13 -2 6 7]) *
            v[2 4 7; 1]

        end
    
    ZZ, vecs, info =
        eigsolve(TensorMap(randn, scalartype(N), space(N, 1) ⊗ V' ⊗ V ← space(N, 1)), 1, :LM) do v
            
            @tensor opt=true vout[-4 -1 -2; -3] := 
            N[1 15 10; -3] *
            Z[-4 3 6; 2] *
            f_north[16 17; 15] *
            conj(f_north[11 12; 10]) *
            Z_virt[11; 14] *
            Z_virt[19; 16] *
            conj(f_north[19 17; 18]) *
            f_north[14 12; 13] *
            ψ_blocked[1, 1][5; 18 -1 3 4] *
            conj(ψ_blocked[1, 1][5; 13 -2 6 7]) *
            v[2 4 7; 1]

        end
    
    # ZIE, vecs, info =
    #     eigsolve(TensorMap(randn, scalartype(N), space(N, 1) ⊗ V' ⊗ V ← space(N, 1)), 1, :LM) do v
            
    #         @tensor opt=true vout[-4 -1 -2; -3] := 
    #         N[1 10 9; -3] *
    #         Z[-4 3 6; 2] *
    #         f_north[11 12; 10] *
    #         Z_virt[14; 11] *
    #         conj(f_north[14 12; 13]) *
    #         ψ_blocked[1, 1][5; 13 20 3 4] *
    #         conj(f_east[21 22; 20]) *
    #         Z_virt[21; 23] *
    #         f_east[23 22; -1] *
    #         conj(ψ_blocked[1, 1][5; 9 -2 6 7]) *
    #         v[2 4 7; 1]

    #     end


    return II, ZI, IZ, ZZ
end

