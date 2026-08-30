#!/usr/bin/env julia
# Multi-GPU gravity inversion on a SYNTHETIC dipping staircase body: N_STEPS
# stacked blocks whose y-window shifts with depth (z), simulating a dipping
# tabular structure. Same flattening convention (nx-fastest), .mat export
# layout, and metrics logging as the other synthetic scripts, so all
# examples land in one comparable CSV.
#
# SCALED-UP VERSION: mesh + observation grid sized so the dense G matrix is
# ~35 GB (nObs * nCells * 4 bytes), matching the same "just under a 40GB A100"
# safety margin used elsewhere -- large enough that multi-GPU sharding
# actually has something to show, and that CPU/1-GPU/4-GPU produce a
# meaningfully different comparison (unlike a small mesh where sync overhead
# dominates any GPU benefit).
#
# Runs on CPU, 1 GPU, or >=2 GPUs automatically -- no hard error, just three
# different result rows (method = cpu_dense / single_gpu / multi_gpu) in the
# same metrics CSV.

cd(@__DIR__)

ENV["GKSwstype"] = "nul"

using LinearAlgebra, SparseArrays, Plots, Printf, CSV, DataFrames, KernelAbstractions, CUDA, Dates
using MAT

gr()
default(colormap = :viridis)

include("../src/GravityInversionGPU.jl")
using .GravityInversionGPU
include("metrics_utils.jl")
using .MetricsUtils

METRICS_CSV = "quantitative_metrics.csv"
EXAMPLE_NAME = "synthetic_dipping_staircase_35GB"

# Number of staircase steps -- easy to scale up/down without touching the
# generation logic below. 8 is a reasonable match for the finer mesh (was 4
# on the smaller 40x40x20 mesh).
N_STEPS = 8

println("\n=== Multi-GPU Gravity Inversion on SYNTHETIC Data (dipping staircase, ~35GB target) ===")
total_start_time = time()
mem_baseline = gpu_mem_used_gb()

# Wrapped in try/catch: calling CUDA.devices() at all when no CUDA driver
# context exists (e.g. a job submitted with no --gres=gpu) throws
# CUDA_ERROR_NOT_INITIALIZED rather than returning 0, which would otherwise
# crash here with a raw stack trace instead of degrading gracefully to the
# CPU-dense path below.
n_visible_gpus = try
    CUDA.functional() ? length(CUDA.devices()) : 0
catch
    0
end
println("Visible CUDA devices: $n_visible_gpus")

# method drives both the metrics CSV row and (implicitly, via
# build_sharded_forward/assign_devices) which devices the sharded algorithm
# actually runs on:
#   0 GPUs visible : "cpu_dense"  -- single CPU "shard" (assign_devices returns [:cpu])
#   1 GPU visible  : "single_gpu" -- single-GPU baseline, no sharding
#   >1 GPU visible : "multi_gpu"  -- sharded across all visible GPUs
if n_visible_gpus == 0
    @warn "No CUDA devices visible -- running as a CPU-DENSE baseline. At ~35GB the " *
          "WHOLE dense G matrix must fit in host RAM as one chunk -- make sure your " *
          "sbatch --mem is comfortably above 35GB before running this on CPU."
elseif n_visible_gpus == 1
    @warn "Only 1 GPU visible -- running as a SINGLE-GPU baseline (no sharding). At " *
          "~35GB this leaves almost no headroom on a 40GB A100 for driver/Julia " *
          "overhead -- if you hit an OOM, this is why. Multi-GPU sharding (>=2 GPUs) " *
          "splits this into ~35/n_gpus GB per device instead."
end
run_method = if n_visible_gpus == 0
    "cpu_dense"
elseif n_visible_gpus == 1
    "single_gpu"
else
    "multi_gpu"
end

# ---- 0. SMOKE TEST: confirm each shard actually lands on its own device ----
# On the CPU-dense path (n_visible_gpus == 0), Gs_test.devices will be
# [:cpu] and Gs_test.chunks[1] a plain Matrix{Float32} -- CUDA.device(...)
# doesn't apply, so the device-placement check only runs when a real GPU
# shard is in play.
println("\n--- Smoke test: tiny 2-point mesh, checking device placement ---")
smoke_mesh = (xm_min=0.0, ym_min=0.0, z0=0.0, dx=100.0, dy=100.0, dz=100.0,
              nx=2, ny=2, nz=2, eps=0.1, delta=1e-4)
smoke_x = [0.0, 100.0, 0.0, 100.0]
smoke_y = [0.0, 0.0, 100.0, 100.0]
Gs_test, _, _ = build_sharded_forward(smoke_mesh, smoke_x, smoke_y, 1e-4)
for (i, dev) in enumerate(Gs_test.devices)
    if dev === :cpu
        println("  shard $i requested dev=:cpu, actual=Matrix on CPU  [OK]")
    else
        actual = CUDA.device(Gs_test.chunks[i])
        status = (Int(actual.handle) == dev) ? "OK" : "MISMATCH -- STOP AND DEBUG"
        println("  shard $i requested dev=$dev, actual=$(actual)  [$status]")
    end
end
println("--- Smoke test done ---\n")

# ---- 1. Mesh -- scaled up 5x in x/y/z resolution vs. the original 40x40x20 ----
# Same domain footprint (20km x 20km x 10km depth), just much finer cells,
# which is what drives nCells up to the point the dense G matrix hits ~35GB.
mesh = (
    xm_min = -10.0, ym_min = -10.0, z0 = 0.0,
    dx = 100.0, dy = 100.0, dz = 100.0,
    nx = 200, ny = 200, nz = 100,
    eps = 0.1, delta = 1e-4
)
println("Mesh: nx=$(mesh.nx), ny=$(mesh.ny), nz=$(mesh.nz), dx=$(mesh.dx), dy=$(mesh.dy), dz=$(mesh.dz)")
write_mesh_UBC(mesh)
nCells = mesh.nx * mesh.ny * mesh.nz
println("Total cells: $nCells")

# ---- 2. Observation grid -- 47x47 = 2209 points across the same domain ----
# Chosen so nObs * nCells * 4 bytes ~= 35 GB (see header comment).
xint = range(0.0, 20000.0, length=47)
yint = range(0.0, 20000.0, length=47)
grid = meshgrid(collect(xint), collect(yint))
xobs, yobs = Float64.(vec(grid.x)), Float64.(vec(grid.y))
nObs_est = length(xobs)

gb_est_total = (nObs_est * nCells * 4) / 1e9
n_shards_est = max(n_visible_gpus, 1)
gb_est_per_gpu = gb_est_total / n_shards_est
println("Estimated dense G: $(nObs_est) x $(nCells) = $(round(gb_est_total, digits=2)) GB; " *
        "per-shard (÷$n_shards_est) ~$(round(gb_est_per_gpu, digits=2)) GB")

# Only a hard error if a single shard would clearly overflow even a 40GB
# card (e.g. running single-GPU at a mesh size meant for many GPUs). No
# equivalent guard for cpu_dense -- that depends on your node's --mem, not
# a fixed GPU card size.
if n_visible_gpus > 0 && gb_est_per_gpu > 38
    error("Estimated per-GPU chunk is ~$(round(gb_est_per_gpu,digits=1)) GB, over the 38GB safety " *
          "margin for a 40GB A100. Reduce the mesh/observation grid, or request more GPUs.")
end

# ---- 3. Build G sharded across all visible GPUs (or as one CPU chunk) ----
build_time = @elapsed begin
    global Gs, Qd_chunks, Dd_chunks = build_sharded_forward(mesh, xobs, yobs, mesh.delta)
end
mem_after_build = gpu_mem_used_gb()
println("G build time: $(round(build_time, digits=2)) s")

# ---- 4. Build the TRUE dipping-staircase model, nx-fastest flattening ----
xc = mesh.xm_min .+ mesh.dx .* (0:(mesh.nx - 1)) .+ mesh.dx / 2
yc = mesh.ym_min .+ mesh.dy .* (0:(mesh.ny - 1)) .+ mesh.dy / 2
zc = mesh.z0     .+ mesh.dz .* (0:(mesh.nz - 1)) .+ mesh.dz / 2

m_true_3d = zeros(Float64, mesh.nx, mesh.ny, mesh.nz)
X3 = zeros(Float64, mesh.nx, mesh.ny, mesh.nz)
Y3 = zeros(Float64, mesh.nx, mesh.ny, mesh.nz)
Z3 = zeros(Float64, mesh.nx, mesh.ny, mesh.nz)

# Generate N_STEPS dipping steps programmatically instead of a hardcoded
# 4-entry list, so the staircase scales with N_STEPS. Fixed x-window
# (8000,12000); y and z windows both shift by a fixed increment per step,
# same dipping-staircase geometry as the original 4-step version.
function generate_staircase_steps(n_steps::Int;
        xr::Tuple{Float64,Float64}=(8000.0, 12000.0),
        y_top::Float64=12000.0, y_step::Float64=1000.0,
        z_top::Float64=7000.0, z_step::Float64=1000.0,
        thickness::Float64=1500.0)
    steps = Vector{Tuple{Tuple{Float64,Float64},Tuple{Float64,Float64},Tuple{Float64,Float64}}}()
    for i in 0:(n_steps - 1)
        y_hi = y_top - i * y_step
        y_lo = y_hi - y_step
        z_hi = z_top - i * z_step
        z_lo = z_hi - thickness
        push!(steps, (xr, (y_lo, y_hi), (z_lo, z_hi)))
    end
    return steps
end

steps = generate_staircase_steps(N_STEPS)
println("Generated $(length(steps)) staircase steps:")
for (i, s) in enumerate(steps)
    println("  step $i: x=$(s[1]), y=$(s[2]), z=$(s[3])")
end

for ix in 1:mesh.nx, iy in 1:mesh.ny, iz in 1:mesh.nz
    x, y, z = xc[ix], yc[iy], zc[iz]
    X3[ix, iy, iz] = x; Y3[ix, iy, iz] = y; Z3[ix, iy, iz] = z

    for (xr, yr, zr) in steps
        if xr[1] < x < xr[2] && yr[1] < y < yr[2] && zr[1] < z < zr[2]
            m_true_3d[ix, iy, iz] = 1.0
            break
        end
    end
end
n_true_cells = count(!iszero, m_true_3d)
println("True dipping-staircase model: $(n_true_cells) / $(nCells) cells set to 1.0")

m_true_flat = vec(Float32.(m_true_3d))
x1 = vec(X3); y1 = vec(Y3); z1 = vec(Z3)
write_model_UBC("model_true_staircase_35GB.start", m_true_flat)

# ---- 5. Forward-model to get synthetic "observed" data ----
d_obs = Array(predict_multi(Gs, m_true_flat))
println("Generated $(length(d_obs)) synthetic gravity observations")

# noise_level must stay Float32: d_obs and the whole GPU pipeline are Float32,
# and Inversion_GPU_multi's obs_host argument requires Vector{Float32}. Using
# a bare Int literal (0) here previously promoted the broadcast result to
# Float64 (Int64 * Float32 -> Float64 under Julia's promotion rules), which
# would have thrown a MethodError at the Inversion_GPU_multi call below.
noise_level = 0.0f0
gravity_data = d_obs .+ noise_level .* randn(Float32, length(d_obs))

p = scatter(xobs, yobs, label="Synthetic observation points", color=:red, markersize=2)
xlabel!(p, "X"); ylabel!(p, "Y"); title!(p, "Synthetic Data Points (Dipping staircase, 35GB run, $run_method)")
savefig(p, "obs_points_staircase_35GB_$(run_method).png")

# ---- 6. Run the inversion (CPU-dense / single-GPU / multi-GPU) ----
t = @elapsed begin
    inverted_model = Inversion_GPU_multi(
        Gs, Qd_chunks, Dd_chunks, gravity_data,
        Float32(mesh.delta), 30, 20; use_squared=true
    )
end
println("Inversion ($run_method) finished in $(round(t, digits=2)) seconds")
mem_after_inversion = gpu_mem_used_gb()
inversion_time = t

# ---- 7. Predicted data ----
inverted_data = predict_multi(Gs, Float32.(inverted_model))
p3 = scatter(xobs, yobs, zcolor=gravity_data, markersize=3, markerstrokewidth=0,
             colorbar_title="mGal", color=:viridis, title="Synthetic 'Observed' (Staircase, 35GB)", xlabel="X", ylabel="Y")
p4 = scatter(xobs, yobs, zcolor=inverted_data, markersize=3, markerstrokewidth=0,
             colorbar_title="mGal", color=:viridis, title="Predicted (Staircase, 35GB, $run_method)", xlabel="X", ylabel="Y")
savefig(plot(p3, p4, layout=(1,2), size=(1000,450)), "observed_vs_predicted_staircase_35GB_$(run_method).png")

data_misfit = norm(inverted_data .- gravity_data) / norm(gravity_data)
println("Relative data misfit = $(round(data_misfit, digits=4))")

# ---- 8. Export .mat files (same variable layout as two_bodies*.mat / staircase_model*.mat) ----
inverted_model_cpu = Array(inverted_model)
matwrite("staircase_model_35GB.mat", Dict("x1"=>x1, "y1"=>y1, "z1"=>z1, "m"=>Float64.(m_true_flat)); compress=true)
matwrite("staircase_model_inversion_35GB_$(run_method).mat", Dict(
    "model"=>Float64.(inverted_model_cpu),
    "nx"=>mesh.nx, "ny"=>mesh.ny, "nz"=>mesh.nz,
    "xm_min"=>mesh.xm_min, "ym_min"=>mesh.ym_min, "z0"=>mesh.z0,
    "dx"=>mesh.dx, "dy"=>mesh.dy, "dz"=>mesh.dz
); compress=true)
println("Saved staircase_model_35GB.mat and staircase_model_inversion_35GB_$(run_method).mat")

# ---- 9. Recovery check ----
model_3d_check = reshape(inverted_model_cpu, (mesh.nx, mesh.ny, mesh.nz))
model_err = norm(vec(model_3d_check) .- vec(m_true_3d)) / norm(vec(m_true_3d))
println("Relative model recovery error = $(round(model_err, digits=4))")

total_time = time() - total_start_time
println("Total runtime: $(round(total_time, digits=2)) seconds")

# ---- 10. Log quantitative metrics (same schema as the other synthetic examples) ----
# Guard against the cpu_dense case where gpu_mem_used_gb() returns an empty
# vector (no functional CUDA device), which would make maximum() throw.
peak_per_gpu = max.(mem_baseline, mem_after_build, mem_after_inversion)
peak_mem_str = isempty(peak_per_gpu) ? "n/a_cpu_dense" : join(round.(peak_per_gpu, digits=2), ";")
peak_mem_max = isempty(peak_per_gpu) ? NaN : round(maximum(peak_per_gpu), digits=2)

row = Dict(
    "timestamp"              => string(Dates.now()),
    "example_name"            => EXAMPLE_NAME,
    "method"                  => run_method,
    "n_gpus"                  => n_visible_gpus,
    "nx"                      => mesh.nx, "ny" => mesh.ny, "nz" => mesh.nz,
    "nCells"                   => nCells,
    "nObs"                     => nObs_est,
    "dense_G_size_GB"          => round(gb_est_total, digits=3),
    "per_gpu_G_size_GB_est"    => round(gb_est_per_gpu, digits=3),
    "G_build_time_s"           => round(build_time, digits=3),
    "inversion_time_s"         => round(inversion_time, digits=3),
    "total_time_s"             => round(total_time, digits=3),
    "peak_gpu_mem_GB_per_device" => peak_mem_str,
    "peak_gpu_mem_GB_max"      => peak_mem_max,
    "itmax"                    => 30, "igmax" => 20,
    "data_misfit_relnorm"      => round(data_misfit, digits=4),
    "model_recovery_error_relnorm" => round(model_err, digits=4),
    "n_steps"                  => N_STEPS,
)
log_metrics!(METRICS_CSV, row)
println("\nLogged row to $METRICS_CSV (method=$run_method). Run this same script on CPU-only, " *
        "1-GPU, and multi-GPU allocations to build the full comparison table.")