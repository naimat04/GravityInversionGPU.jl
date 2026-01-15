#!/usr/bin/env julia
# Example script for running gravity inversion

cd(@__DIR__)

# Force GR into offscreen mode
ENV["GKSwstype"] = "nul"

# Load packages needed for this script
using LinearAlgebra, SparseArrays, Plots, Printf, ArgParse, KernelAbstractions

# Set up plotting
gr()
default(colormap = :viridis)

# Load our module
include("../src/GravityInversionGPU.jl")
using .GravityInversionGPU

function test_gravity_response_GPU(args)
    println("\n=== Starting GPU-Accelerated Gravity Inversion ===")
    total_start_time = time()
    
    # Parse command-line arguments
    s = ArgParseSettings()
    @add_arg_table s begin
        "--nx"
            help = "Number of grid points in x-direction"
            arg_type = Int
            default = 40

        "--ny"
            help = "Number of grid points in y-direction"
            arg_type = Int
            default = 40

        "--nz"
            help = "Number of grid points in z-direction"
            arg_type = Int
            default = 20
    end

    parsed_args = parse_args(args, s)
    nx, ny, nz = parsed_args["nx"], parsed_args["ny"], parsed_args["nz"]
    println("Using mesh dimensions: nx=$nx, ny=$ny, nz=$nz")
    println("Using backend: $BACKEND")
    
    # Define mesh with the provided parameters
    mesh = (
        xm_min = -20.0, ym_min = -20.0, z0 = 0.0,
        dx = 500.0, dy = 500.0, dz = 500.0,
        nx = nx, ny = ny, nz = nz,
        eps = 0.1, delta = 1e-4
    )

    write_mesh_UBC(mesh)
    
    # Create true model (CPU)
    nCells = mesh.nx * mesh.ny * mesh.nz
    model_true = zeros(Float64, nCells)
    xc = [mesh.xm_min + (i-0.5)*mesh.dx for i in 1:mesh.nx, j in 1:mesh.ny, k in 1:mesh.nz]
    yc = [mesh.ym_min + (j-0.5)*mesh.dy for i in 1:mesh.nx, j in 1:mesh.ny, k in 1:mesh.nz]
    zc = [(k-0.5)*mesh.dz for i in 1:mesh.nx, j in 1:mesh.ny, k in 1:mesh.nz]
    
    for idx in 1:nCells
        if (8000 < xc[idx] < 12000 && 8000 < yc[idx] < 12000 && 4000 < zc[idx] < 6000) ||
           (1000 < xc[idx] < 4000 && 1000 < yc[idx] < 5000 && 2000 < zc[idx] < 6000)
            model_true[idx] = 0.1
        end
    end
    
    # Write models
    write_model_UBC("model_gpu.true", model_true)
    write_model_UBC("model_gpu.start", fill(0.0001, nCells))
    
    # Define observation grid (can be customized by user)
    x_obs_range = 0:mesh.dx:19000  # From 0 to 19km with mesh.dx spacing
    y_obs_range = 0:mesh.dy:19000  # From 0 to 19km with mesh.dy spacing
    
    # Generate synthetic gravity data on GPU
    println("\nGenerating synthetic gravity data...")
    println("Observation grid: X = $(minimum(x_obs_range)):$(mesh.dx):$(maximum(x_obs_range)) m")
    println("                  Y = $(minimum(y_obs_range)):$(mesh.dy):$(maximum(y_obs_range)) m")
    
    gravity_data, G_gpu, Q_diag_gpu, D_diag_gpu, xobs, yobs, x1, y1, z1 = 
        Gravity_response3D_GPU(model_true, mesh, x_obs_range, y_obs_range)
    
    println("Generated $(length(gravity_data)) gravity observations")
    
    # Run inversion on GPU
    obs_gpu = KernelAbstractions.allocate(BACKEND, Float64, length(gravity_data))
    copyto!(obs_gpu, gravity_data)
    
    inverted_model = Inversion_GPU(
        G_gpu, Q_diag_gpu, D_diag_gpu, obs_gpu, 
        mesh.delta, 10, 20
    )
    
    # Save results
    write_model_UBC("model_gpu.inv", inverted_model)
    
    # Compute predicted data
    inv_model_ka = KernelAbstractions.allocate(BACKEND, Float64, length(inverted_model))
    copyto!(inv_model_ka, inverted_model)
    tmp_result = similar(inv_model_ka, size(G_gpu, 1))
    mul!(tmp_result, G_gpu, inv_model_ka)
    inverted_data = Array(tmp_result)
    
    write_data_UBC("data_gpu.pred", xobs, yobs, inverted_data, mesh.delta)
    
    # Generate plots
    composite_surface_plot(xobs, yobs, gravity_data, inverted_data)
    composite_model_plot(model_true, inverted_model, mesh)
    
    total_time = time() - total_start_time
    println("=== GPU Inversion Complete ===")
    println("Total execution time: $(total_time) seconds")
end

# ALWAYS RUN - remove the if condition
println("Script starting...")
test_gravity_response_GPU(ARGS)