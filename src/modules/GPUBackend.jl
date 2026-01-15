# GPU backend detection and management
using KernelAbstractions

function detect_best_backend()
    # Try backends in order of preference
    backend_order = [
        (:Metal, :MetalBackend, :functional),      # Apple Silicon
        (:CUDA, :CUDABackend, :functional),        # NVIDIA
        (:AMDGPU, :ROCBackend, :functional),       # AMD
        (:oneAPI, :oneAPIBackend, :functional),    # Intel
    ]
    
    for (pkg_name, backend_type, check_func) in backend_order
        try
            @eval using $pkg_name
            if @eval $pkg_name.$check_func()
                println("✓ Using $(string(pkg_name)) backend")
                return @eval $pkg_name.$backend_type()
            end
        catch e
            println("  $(string(pkg_name)) not available: $e")
            continue  # Try next backend
        end
    end
    
    println("⚠ No GPU found, using CPU backend")
    return CPU()
end

# Global backend variable
const BACKEND = detect_best_backend()
println("Selected backend: $BACKEND")
