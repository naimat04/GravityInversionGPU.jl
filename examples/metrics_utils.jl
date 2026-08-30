module MetricsUtils

using CUDA, CSV, DataFrames, Dates

export gpu_mem_used_gb, log_metrics!

"""
    gpu_mem_used_gb() -> Vector{Float64}

Used memory (GB) on every visible CUDA device, in device order.
Call this at several checkpoints (baseline, after G build, after inversion)
and take an elementwise max to get a peak-usage snapshot per device.

CUDA.jl has renamed the free/total memory query functions across versions
(`available_memory`/`total_memory` in older releases, `free_memory` in
newer ones). This tries each known name so the same script works regardless
of which CUDA.jl version is installed on Mahti, and falls back to NaN
(with a one-time warning) rather than crashing the whole run if none work.
"""
function _free_total_bytes()
    if isdefined(CUDA, :available_memory) && isdefined(CUDA, :total_memory)
        return CUDA.available_memory(), CUDA.total_memory()
    elseif isdefined(CUDA, :free_memory) && isdefined(CUDA, :total_memory)
        return CUDA.free_memory(), CUDA.total_memory()
    elseif isdefined(CUDA, :memory_info)
        info = CUDA.memory_info()  # some versions return (free, total)
        return info[1], info[2]
    else
        return nothing
    end
end

const _WARNED_NO_MEM_API = Ref(false)
function gpu_mem_used_gb()
    used = Float64[]
    if !CUDA.functional()
        return used
    end
    for dev in CUDA.devices()
        CUDA.device!(dev)
        result = _free_total_bytes()
        if result === nothing
            if !_WARNED_NO_MEM_API[]
                @warn "No known CUDA.jl memory-query function found (tried available_memory, free_memory, memory_info). Memory columns in the metrics CSV will be NaN. Check `names(CUDA)` on this system to find the right one."
                _WARNED_NO_MEM_API[] = true
            end
            push!(used, NaN)
        else
            free_b, total_b = result
            push!(used, (total_b - free_b) / 1e9)
        end
    end
    return used
end

"""
    log_metrics!(csv_path, row::Dict)

Append one row of quantitative metrics to `csv_path`, creating the file
with a header on first use. Designed so multiple methods (multi_gpu,
single_gpu, cpu_dense) and multiple examples can be appended over separate
runs and later loaded together as one comparison table (e.g. in Python/
pandas or MATLAB) for the manuscript revision.
"""
function log_metrics!(csv_path::AbstractString, row::Dict)
    df_row = DataFrame([row])
    if isfile(csv_path)
        # Align columns in case a later run adds new fields; missing values
        # for older/newer columns are filled with `missing`.
        existing = CSV.read(csv_path, DataFrame)
        combined = vcat(existing, df_row; cols=:union)
        CSV.write(csv_path, combined)
    else
        CSV.write(csv_path, df_row)
    end
    println("Logged metrics row to $csv_path (method=$(get(row,"method","?")), example=$(get(row,"example_name","?")))")
end

end # module