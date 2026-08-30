# Multi-GPU row-sharded gravity inversion
#
# ── HOW THIS WORKS ──────────────────────────────────────────────────────────
# The dense sensitivity matrix G is (nObs x nCells). Observations are
# independent of each other in the forward operator, so G can be split by
# ROW (observation) across N GPUs with no change to the underlying physics:
#   G = [ G_1 ]   <- lives entirely on GPU 0
#       [ G_2 ]   <- lives entirely on GPU 1
#       [ ... ]
#       [ G_N ]   <- lives entirely on GPU N-1
#
# Looking at the existing single-GPU CG_GPU/Inversion_GPU code, the vectors
# that are naturally "observation-space" (length nObs) are: f, r0, y0, p0,
# x0, Ap, D_diag, M_diag, obs. These can stay row-sharded on their own GPU
# for the entire algorithm.
#
# The vectors that are "model-space" (length nCells) are: mk, Sdiag, Q_diag,
# and the intermediate `tmp`/`tmp2` inside CG_GPU. These must be REPLICATED
# (same values on every GPU) so each GPU can do its local elementwise ops.
#
# The only place data has to move between GPUs is:
#   tmp = A' * p0   (A' * p0 is a REDUCTION over observations -> needs an
#                     all-reduce: each GPU computes its own partial nCells-
#                     length result, then all partials are summed and the
#                     total is copied back out to every GPU)
#
# Everything else (A * x, elementwise ops, dot products) needs no cross-GPU
# communication beyond a final scalar sum for dot products.
#
# ── mk.^2 PARAMETERIZATION IS NOW A RUNTIME SWITCH ──────────────────────────
# The original algorithm parameterizes the model as m = mk^2 (a standard
# trick to force positivity). Previously this was toggled by commenting /
# uncommenting three separate lines by hand, which is error-prone (see the
# note below). It is now controlled by a single `use_squared::Bool` argument
# passed into `Inversion_GPU_multi` (or supplied interactively — see
# `ask_use_squared()` below). Internally this flows through a tiny pair of
# closures, `mk2_of(mk)` and `sdiag_of(mk)`, so the three dependent spots
# always stay consistent with each other:
#
#   1. mk2_chunks   = mk2_of(mk)     -> mk^2 if use_squared else mk
#   2. Sdiag update = sdiag_of(mk)   -> 2*mk if use_squared else identically 1
#                                        (Sdiag is d(m)/d(mk); when m = mk,
#                                        this derivative is 1, not mk)
#   3. final return = mk2_of(mk)     -> squares (or doesn't) on the way out
#
# Because all three read the same `use_squared` flag, it is no longer
# possible to end up in the old silent-bug hybrid state (squaring mk2 but
# forgetting to also update Sdiag, or vice versa).
#
# ── IMPORTANT CORRECTNESS REQUIREMENT ───────────────────────────────────────
# Every operation touching a CuArray on a particular device MUST run while
# that device is the "active" CUDA device for the current task (CUDA.jl uses
# a task-local current-device pointer). This file wraps every allocation and
# op in `with_device(dev) do ... end` for that reason (see the GPU/CPU
# auto-detection section below) -- on a real GPU that's exactly the old
# `CUDA.device!(dev) do ... end`; on the `:cpu` sentinel it's a no-op wrapper
# with no CUDA calls at all. Do not remove those wrappers when editing this
# file, even for what looks like a harmless elementwise `.*` — using the
# wrong active device against another device's array either errors
# immediately (no P2P) or silently corrupts results (P2P enabled), depending
# on cluster config.
#
# ── GPU / CPU: NO SEPARATE CODE PATH NEEDED ─────────────────────────────────
# assign_devices() / build_sharded_forward() auto-detect how many CUDA GPUs
# are actually visible to this job:
#   >=1 CUDA GPU visible : shards across that many real devices, as before.
#   0 CUDA GPUs visible  : falls back to a single CPU "shard" (devices=[:cpu]),
#                          running the exact same sharded algorithm -- just
#                          with one chunk holding the whole problem, on plain
#                          Arrays instead of CuArrays. Nothing about the
#                          CG_GPU_multi / Inversion_GPU_multi algorithm itself
#                          changes between the two cases.
# This means the SAME script, run under an sbatch allocation with 4 GPUs, 1
# GPU, or 0 GPUs, automatically does the right thing -- useful for producing
# the multi_gpu / single_gpu / cpu rows of a compute+memory comparison table
# without maintaining three versions of the driver script.
#
# ── SMOKE TEST BEFORE A FULL RUN ────────────────────────────────────────────
# Before trusting a large run, run build_sharded_forward on a tiny grid and
# check that each chunk actually landed on the right device:
#   for (i, dev) in enumerate(Gs.devices)
#       println("chunk $i requested dev=$dev, actual=$(CUDA.device(Gs.chunks[i]))")
#   end
# All lines should match. If any mismatch, do not proceed to a full run.

# ── GPU / CPU AUTO-DETECTION ─────────────────────────────────────────────────
# Every function below used to hardcode CUDA: `devices::Vector{Int}` (CUDA
# device ids), `CUDA.device!(dev) do ... end` around every op, and `CuArray(...)`
# for allocation. That's now generalized to a "device tag" that is EITHER a
# real CUDA device id (Int) OR the sentinel `:cpu`, meaning "run this chunk on
# the CPU instead". The whole sharded algorithm (chunk_map, dot_multi,
# CG_GPU_multi, Inversion_GPU_multi, ...) is unchanged in structure -- it just
# also works when devs == [:cpu] and nchunks == 1, which is the "no CUDA GPU
# visible" case, running on plain Arrays instead of CuArrays.
#
# `with_device(dev) do ... end` replaces every old `CUDA.device!(dev) do ... end`:
# on a real GPU it's identical to before; on :cpu it just runs the block with
# no CUDA calls at all (important, since even calling CUDA.device! when CUDA's
# driver isn't initialized can throw).
#
# `alloc_on(dev, v_host)` replaces the old bare `CuArray(v_host)`: CuArray on
# a real GPU, plain Array on :cpu.
#
# Nothing about the field-inversion / positivity logic below changes; this
# section only decides WHERE the arrays for the chosen parameterization live.

is_gpu_dev(dev) = dev isa Integer

"""
    detect_gpu_count()

Number of CUDA GPUs actually visible to this job. Wrapped in try/catch
because calling `CUDA.devices()` (or anything else CUDA-related) when the
CUDA driver hasn't been initialized -- e.g. a CPU-only sbatch allocation
with no --gres=gpu -- throws CUDA_ERROR_NOT_INITIALIZED rather than just
returning 0. Any failure here is treated as "no GPU available".
"""
function detect_gpu_count()
    try
        return CUDA.functional() ? Int(length(CUDA.devices())) : 0
    catch
        return 0
    end
end

"""
    with_device(f, dev)

Run `f()` with `dev` as the active CUDA device (if `dev` is a real GPU id),
or just run `f()` directly with no CUDA calls at all (if `dev === :cpu`).
Drop-in replacement for every old `CUDA.device!(dev) do ... end` block.
"""
function with_device(f, dev)
    if is_gpu_dev(dev)
        return CUDA.device!(f, dev)
    else
        return f()
    end
end

"""
    alloc_on(dev, v_host::Vector{Float32})

Allocate a device array on `dev` and copy `v_host` into it: `CuArray` for a
real GPU, plain `Array` (i.e. just `v_host` itself, no device transfer) for
`:cpu`. Drop-in replacement for the old bare `CuArray(v_host)` calls.
"""
alloc_on(dev, v_host::Vector{Float32}) = is_gpu_dev(dev) ? CuArray(v_host) : Array(v_host)

using CUDA

struct ShardedMatrix
    chunks::Vector{Any}              # G_i :: CuArray{Float32,2} (GPU) or Matrix{Float32} (CPU), one per shard
    devices::Vector{Any}             # CUDA device ids (Int, 0-indexed) or :cpu
    row_ranges::Vector{UnitRange{Int}}
    nobs::Int
    ncells::Int
end

"""
    assign_devices(nchunks=nothing)

Pick `nchunks` distinct CUDA device ids (0-indexed) if any GPUs are visible,
erroring loudly if the job doesn't actually have that many GPUs (e.g. your
sbatch --gres=gpu:a100:N line doesn't match what you asked for). If NO CUDA
GPUs are visible at all, returns `[:cpu]` (a single CPU "shard") regardless
of what `nchunks` was asked for, so the whole pipeline degrades gracefully
to a single-device CPU run instead of erroring.
"""
function assign_devices(nchunks::Union{Integer,Nothing}=nothing)
    ngpu = detect_gpu_count()
    if ngpu == 0
        if nchunks !== nothing && nchunks > 1
            @warn "Requested $nchunks GPU shards but no CUDA GPU is visible in this job -- " *
                  "running as a single CPU chunk instead."
        end
        return Any[:cpu]
    end
    n = nchunks === nothing ? ngpu : Int(nchunks)
    if n > ngpu
        error("Requested $n GPU shards but only $ngpu CUDA device(s) are visible in this job. " *
              "Check your sbatch --gres=gpu:a100:N line matches nchunks.")
    end
    return Any[i for i in 0:(n - 1)]
end

"""
    partition_rows(n, nchunks)

Balanced (as-equal-as-possible) partition of 1:n into `nchunks` contiguous
UnitRanges, used to split observations across GPUs.
"""
function partition_rows(n::Integer, nchunks::Integer)
    n, nchunks = Int(n), Int(nchunks)
    base = div(n, nchunks)
    rem_ = n % nchunks
    ranges = UnitRange{Int}[]
    start = 1
    for i in 1:nchunks
        len = base + (i <= rem_ ? 1 : 0)
        push!(ranges, start:(start + len - 1))
        start += len
    end
    return ranges
end

"""
    replicate_to_devices(v_host, devices)

Copy the same host vector to every listed device. Used for model-space
vectors (mk, Sdiag, Q_diag, and the all-reduced `tmp` from A'*p) that every
GPU needs an identical local copy of.
"""
function replicate_to_devices(v_host::Vector{Float32}, devices::Vector)
    out = Vector{Any}(undef, length(devices))
    for (i, dev) in enumerate(devices)
        with_device(dev) do
            out[i] = alloc_on(dev, v_host)
        end
    end
    return out
end

"""
    scatter_to_devices(v_host, row_ranges, devices)

Split a host vector by row_ranges and send each slice to its matching
device. Used for observation-space vectors (obs, D_diag) that are naturally
row-sharded.
"""
function scatter_to_devices(v_host::AbstractVector, row_ranges::Vector{UnitRange{Int}}, devices::Vector)
    out = Vector{Any}(undef, length(devices))
    for (i, dev) in enumerate(devices)
        with_device(dev) do
            out[i] = alloc_on(dev, Float32.(v_host[row_ranges[i]]))
        end
    end
    return out
end

"""
    gather_from_devices(chunks, row_ranges, n)

Concatenate row-sharded device chunks back into one host vector of length n.
Only needed when you actually want the full vector on the CPU (e.g. for
plotting/saving), not during the CG iterations themselves.
"""
function gather_from_devices(chunks::Vector{Any}, row_ranges::Vector{UnitRange{Int}}, n::Int)
    out = Vector{Float32}(undef, n)
    for (i, c) in enumerate(chunks)
        out[row_ranges[i]] = Array(c)
    end
    return out
end

"""
    chunk_map(f, devs, chunks...)

Apply `f` elementwise across matching per-device chunks, running each
device's computation while that device is active. `chunks...` are each
Vector{Any} of length(devs) (e.g. p0, r0). Returns a new Vector{Any} of
the same shape.

Example: `r0 = chunk_map(fi -> -fi, devs, f_chunks)`
Example: `y0 = chunk_map((m,r) -> m .* r, devs, M_chunks, r0)`
"""
function chunk_map(f, devs::Vector, chunks::Vector{Any}...)
    n = length(devs)
    out = Vector{Any}(undef, n)
    for i in 1:n
        with_device(devs[i]) do
            out[i] = f(ntuple(j -> chunks[j][i], length(chunks))...)
        end
    end
    return out
end

"""
    dot_multi(a_chunks, b_chunks)

Global dot product across row-sharded vectors: sum each device's local dot
product (reusing backend_dot from CoreFunctions.jl), then sum those partial
scalars on the host. Scalars are cheap to move, no explicit device! needed
for the final host-side sum.
"""
function dot_multi(a_chunks::Vector{Any}, b_chunks::Vector{Any}, devs::Vector)
    s = 0.0f0
    for i in eachindex(a_chunks)
        with_device(devs[i]) do
            s += Float32(backend_dot(a_chunks[i], b_chunks[i]))
        end
    end
    return s
end

"""
    mul_forward_multi(Gs, x_chunks)

y = A * x, where x is a REPLICATED model-space vector (same values on every
device). No communication needed: each device multiplies its own row-chunk
of G by its local copy of x, producing that device's slice of y. Returns
row-sharded chunks (not gathered).
"""
function mul_forward_multi(Gs::ShardedMatrix, x_chunks::Vector{Any})
    out = Vector{Any}(undef, length(Gs.chunks))
    for (i, dev) in enumerate(Gs.devices)
        with_device(dev) do
            out[i] = Gs.chunks[i] * x_chunks[i]
        end
    end
    return out
end

"""
    mul_transpose_multi(Gs, p_chunks)

tmp = A' * p, where p is row-sharded (data-space). This is the one place
that needs real cross-GPU communication: each device computes its own
partial nCells-length result from its row-chunk, the partials are summed
on the host (all-reduce), and the total is copied back out to every device
so subsequent elementwise ops (tmp2 = SQS_diag .* tmp) can run locally.
"""
function mul_transpose_multi(Gs::ShardedMatrix, p_chunks::Vector{Any})
    partials = Vector{Vector{Float32}}(undef, length(Gs.chunks))
    for (i, dev) in enumerate(Gs.devices)
        with_device(dev) do
            partials[i] = Array(Gs.chunks[i]' * p_chunks[i])
        end
    end
    total = reduce(+, partials)  # host-side all-reduce, length ncells
    return replicate_to_devices(total, Gs.devices)
end

function build_sharded_forward(mesh::NamedTuple, xobs::Vector{Float64}, yobs::Vector{Float64},
                                delta::Float64; nchunks::Union{Integer,Nothing}=nothing,
                                devices::Union{Vector,Nothing}=nothing)
    n = length(xobs)
    devs = devices !== nothing ? devices : assign_devices(nchunks)
    ndevs = length(devs)
    row_ranges = partition_rows(n, ndevs)

    chunks = Vector{Any}(undef, ndevs)
    Qd_chunks = Vector{Any}(undef, ndevs)
    Dd_chunks = Vector{Any}(undef, ndevs)
    ncells_val = Ref(0)

    label = is_gpu_dev(devs[1]) ? "GPU" : "CPU"
    println("Building sharded G across $ndevs $label shard(s):")
    for (i, dev) in enumerate(devs)
        rows = row_ranges[i]
        with_device(dev) do
            G_i, Qd_i, Dd_i, _, _, _ = Call_matrix_KA(
                mesh.xm_min, mesh.ym_min, xobs[rows], yobs[rows], mesh.z0,
                mesh.dx, mesh.dy, mesh.dz, mesh.nx, mesh.ny, mesh.nz,
                mesh.eps, delta; backend=BACKEND
            )
            chunks[i] = G_i
            Qd_chunks[i] = Qd_i
            Dd_chunks[i] = Dd_i
            ncells_val[] = size(G_i, 2)
            gb = (sizeof(Float32) * size(G_i, 1) * size(G_i, 2)) / 1e9
            println("  [$label $dev] rows $(rows) -> chunk $(size(G_i)) ($(round(gb, digits=2)) GB)")
        end
    end

    Gs = ShardedMatrix(chunks, devs, row_ranges, n, ncells_val[])
    total_gb = sum((sizeof(Float32) * size(c, 1) * size(c, 2)) / 1e9 for c in chunks)
    println("Total G memory across all shard(s): $(round(total_gb, digits=2)) GB")
    return Gs, Qd_chunks, Dd_chunks
end

"""
    CG_GPU_multi(Gs, SQS_chunks, Dd_chunks, f_chunks, M_chunks, igmax, delta)

Multi-GPU version of CG_GPU. Same algorithm as the single-GPU version in
Inversion.jl, but every observation-space vector (r0, y0, p0, x0, Ap) stays
row-sharded across devices, and the one model-space reduction (A' * p0)
goes through mul_transpose_multi's all-reduce.

Unaffected by the mk / mk.^2 switch — SQS_chunks already has whatever
parameterization was chosen baked in by the caller.
"""
function CG_GPU_multi(Gs::ShardedMatrix, SQS_chunks::Vector{Any}, Dd_chunks::Vector{Any},
                       f_chunks::Vector{Any}, M_chunks::Vector{Any}, igmax::Int, delta::Float32)
    devs = Gs.devices

    r0 = chunk_map(fi -> -fi, devs, f_chunks)
    y0 = chunk_map((m, r) -> m .* r, devs, M_chunks, r0)
    p0 = chunk_map(yi -> -yi, devs, y0)
    x0 = Vector{Any}(undef, length(devs))
    for (i, dev) in enumerate(devs)
        with_device(dev) do
            x0[i] = KernelAbstractions.zeros(BACKEND, Float32, length(r0[i]))
        end
    end

    k = 0
    res = dot_multi(r0, r0, devs) / Gs.nobs
    while k < igmax && res > delta
        tmp_rep = mul_transpose_multi(Gs, p0)
        tmp2 = chunk_map((s, t) -> s .* t, devs, SQS_chunks, tmp_rep)
        Ap1 = mul_forward_multi(Gs, tmp2)
        Ap2 = chunk_map((d, p) -> d .* p, devs, Dd_chunks, p0)
        Ap = chunk_map((a1, a2) -> a1 .+ a2, devs, Ap1, Ap2)

        numerator = dot_multi(r0, y0, devs)
        denominator = dot_multi(p0, Ap, devs)
        a0 = numerator / denominator

        x0 = chunk_map((x, p) -> x .+ a0 .* p, devs, x0, p0)
        r1 = chunk_map((r, a) -> r .+ a0 .* a, devs, r0, Ap)
        y1 = chunk_map((m, r) -> m .* r, devs, M_chunks, r1)
        b0 = dot_multi(r1, y1, devs) / numerator
        p0 = chunk_map((y, p) -> -y .+ b0 .* p, devs, y1, p0)

        res = dot_multi(r1, r1, devs) / Gs.nobs
        r0, y0 = r1, y1
        k += 1
    end
    return x0
end

"""
    ask_use_squared() -> Bool

Interactively prompt on stdin for which parameterization to use this run.
Accepts y/n/yes/no/mk/mk2 (case-insensitive); re-prompts on anything else.
Call this yourself and pass the result into `Inversion_GPU_multi(...;
use_squared=...)`, or just pass `true`/`false` directly if you're scripting
and don't want an interactive prompt.
"""
function ask_use_squared()
    while true
        print("Use m = mk.^2 parameterization? [y]es (mk.^2) / [n]o (mk directly): ")
        ans = lowercase(strip(readline()))
        if ans in ("y", "yes", "mk2", "mk^2", "mk.^2", "true")
            return true
        elseif ans in ("n", "no", "mk", "false")
            return false
        else
            println("  please answer y/n")
        end
    end
end

"""
    Inversion_GPU_multi(Gs, Qd_chunks, Dd_chunks, obs_host, delta, itmax, igmax;
                         use_squared)

Multi-GPU version of Inversion_GPU. `obs_host` is a plain Float32 Vector on
the CPU (length Gs.nobs) — it gets scattered across devices internally.
Returns the inverted model as a plain Float32 Vector on the CPU
(length Gs.ncells), same as the single-GPU version's return type.

`use_squared::Bool` (required keyword, no default on purpose — see below)
picks the model parameterization for THIS call:
  - `use_squared = true`  -> m = mk.^2   (forces positivity; original algorithm)
  - `use_squared = false` -> m = mk      (no positivity constraint)

This replaces the old approach of commenting/uncommenting three separate
lines by hand. All three dependent spots now read the same flag through two
small closures defined at the top of the function body:
  - `mk2_of(mk)`    : the model itself (mk.^2 or mk)
  - `sdiag_of(mk)`  : d(m)/d(mk) — 2*mk when squared, identically 1 when not

There is deliberately no default value for `use_squared`. If you want the
program to ask you every time instead of hard-coding it at each call site,
pass `use_squared = ask_use_squared()`.
"""
function Inversion_GPU_multi(Gs::ShardedMatrix, Qd_chunks::Vector{Any}, Dd_chunks::Vector{Any},
                              obs_host::Vector{Float32}, delta::Float32, itmax::Int, igmax::Int;
                              use_squared::Bool)
    devs = Gs.devices
    m = Gs.ncells
    n = Gs.nobs

    # --- the entire mk vs mk.^2 choice lives in these two closures ---
    mk2_of(mk)   = use_squared ? mk .^ 2 : mk
    sdiag_of(mk) = use_squared ? 2 .* mk : one.(mk)
    param_name = use_squared ? "m = mk.^2" : "m = mk"
    # -------------------------------------------------------------------

    obs_chunks = scatter_to_devices(obs_host, Gs.row_ranges, devs)

    mk_chunks = replicate_to_devices(fill(0.0001f0, m), devs)
    mk2_chunks = chunk_map(mk2_of, devs, mk_chunks)

    Gm_chunks = mul_forward_multi(Gs, mk2_chunks)
    dres_chunks = chunk_map((o, g) -> o .- g, devs, obs_chunks, Gm_chunks)
    res = dot_multi(dres_chunks, dres_chunks, devs) / n

    Sdiag_chunks = replicate_to_devices(ones(Float32, m), devs)
    k = 0
    log_lines = ["Multi-GPU Inversion ($(length(devs)) GPUs, $param_name): itmax=$itmax, igmax=$igmax, initial residual=$res"]

    while k < itmax && res > delta
        SQS_chunks = chunk_map((s, q) -> s .* q .* s, devs, Sdiag_chunks, Qd_chunks)

        mk2_chunks = chunk_map(mk2_of, devs, mk_chunks)

        Gm_chunks = mul_forward_multi(Gs, mk2_chunks)
        f_chunks = chunk_map((o, g) -> o .- g, devs, obs_chunks, Gm_chunks)
        push!(log_lines, "Iter $k: res=$res")

        M_chunks = Vector{Any}(undef, length(devs))
        for (i, dev) in enumerate(devs)
            with_device(dev) do
                M_chunks[i] = KernelAbstractions.ones(BACKEND, Float32, length(Gs.row_ranges[i]))
            end
        end

        x0 = CG_GPU_multi(Gs, SQS_chunks, Dd_chunks, f_chunks, M_chunks, igmax, delta)

        m0_chunks = chunk_map(mk -> copy(mk), devs, mk_chunks)
        Gtx0_rep = mul_transpose_multi(Gs, x0)
        dm_chunks = chunk_map((q, s, g) -> q .* (s .* g), devs, Qd_chunks, Sdiag_chunks, Gtx0_rep)
        mk_chunks = chunk_map((m0, dm) -> m0 .+ dm, devs, m0_chunks, dm_chunks)

        mk2_chunks = chunk_map(mk2_of, devs, mk_chunks)

        Gm1_chunks = mul_forward_multi(Gs, mk2_chunks)
        dres1_chunks = chunk_map((o, g) -> o .- g, devs, obs_chunks, Gm1_chunks)
        res1 = dot_multi(dres1_chunks, dres1_chunks, devs) / n

        while res1 > res
            dm_chunks = chunk_map(dm -> dm ./ 3.0f0, devs, dm_chunks)
            mk_chunks = chunk_map((m0, dm) -> m0 .+ dm, devs, m0_chunks, dm_chunks)
            mk2_chunks = chunk_map(mk2_of, devs, mk_chunks)

            Gm1_chunks = mul_forward_multi(Gs, mk2_chunks)
            dres1_chunks = chunk_map((o, g) -> o .- g, devs, obs_chunks, Gm1_chunks)
            res1 = dot_multi(dres1_chunks, dres1_chunks, devs) / n
            push!(log_lines, "  Step adjust: new res=$res1")
        end

        # Sdiag is d(m)/d(mk). Squared: 2*mk. Unsquared: identically 1.
        # sdiag_of() encodes both cases, so this always matches mk2_of()
        # above — no way to end up with the old mismatched hybrid state.
        Sdiag_chunks = chunk_map(sdiag_of, devs, mk_chunks)

        res = res1
        k += 1
    end

    open("inversion_multigpu.log", "w") do io
        foreach(line -> println(io, line), log_lines)
    end

    # mk is replicated identically on every device (every update was built from
    # already-replicated inputs), so any single device's copy is the full answer.
    # mk2_of() can involve a real kernel launch (.^2 when use_squared=true), not
    # just a memory copy, so — per this file's own device-active invariant — it
    # must run while mk_chunks[1]'s own device is active, not whatever device
    # happened to be left active by the last loop iteration.
    return with_device(devs[1]) do
        Array(mk2_of(mk_chunks[1]))
    end
end

"""
    predict_multi(Gs, model_host)

Forward pass G*model for a finished model, gathered back to one host
vector (length Gs.nobs). Handy for the observed-vs-predicted comparison
plot, mirroring what the single-GPU driver script does.
"""
function predict_multi(Gs::ShardedMatrix, model_host::Vector{Float32})
    m_chunks = replicate_to_devices(model_host, Gs.devices)
    y_chunks = mul_forward_multi(Gs, m_chunks)
    return gather_from_devices(y_chunks, Gs.row_ranges, Gs.nobs)
end