# Inversion algorithms
using LinearAlgebra 
function CG_GPU(A, SQS_diag, D_diag, f, M_diag, igmax::Int, delta::Float64)
    """
    Conjugate Gradient solver for GPU arrays.
    
    Parameters:
    -----------
    A : DeviceArray
        Sensitivity matrix
    SQS_diag, D_diag, M_diag : DeviceArray
        Diagonal preconditioners
    f : DeviceArray
        Misfit vector
    igmax : Int
        Maximum iterations
    delta : Float64
        Convergence tolerance
        
    Returns:
    --------
    DeviceArray : Solution vector
    """
    n = length(f)
    r0 = -f
    y0 = M_diag .* r0
    p0 = -y0
    x0 = KernelAbstractions.zeros(BACKEND, Float64, n)
    
    k = 0
    res = backend_dot(r0, r0) / n
    while k < igmax && res > delta
        tmp = A' * p0
        tmp2 = SQS_diag .* tmp
        Ap1 = A * tmp2
        Ap2 = D_diag .* p0
        Ap = Ap1 .+ Ap2
        
        numerator = backend_dot(r0, y0)
        denominator = backend_dot(p0, Ap)
        a0 = numerator / denominator
        
        x0 .+= a0 .* p0
        r1 = r0 .+ a0 .* Ap
        y1 = M_diag .* r1
        b0 = backend_dot(r1, y1) / numerator
        p0 .= -y1 .+ b0 .* p0
        
        res = backend_dot(r1, r1) / n
        r0, y0 = r1, y1
        k += 1
    end
    return x0
end

function Inversion_GPU(G, Q_diag, D_diag, obs, delta::Float64, itmax::Int, igmax::Int)
    """
    Main inversion algorithm using GPU acceleration.
    
    Parameters:
    -----------
    G : DeviceArray
        Sensitivity matrix
    Q_diag, D_diag : DeviceArray
        Regularization matrices
    obs : DeviceArray
        Observed data
    delta : Float64
        Convergence tolerance
    itmax : Int
        Maximum outer iterations
    igmax : Int
        Maximum inner (CG) iterations
        
    Returns:
    --------
    Vector{Float64} : Inverted model
    """
    n, m = size(G)
    mk = KernelAbstractions.ones(BACKEND, Float64, m) .* 0.0001f0
    dres = obs .- G * (mk .^ 2)
    res = backend_dot(dres, dres) / n
    
    Sdiag = KernelAbstractions.ones(BACKEND, Float64, m)
    k = 0
    log_lines = ["GPU Inversion: itmax=$itmax, igmax=$igmax, initial residual=$res"]
    
    while k < itmax && res > delta
        SQS_diag = Sdiag .* Q_diag .* Sdiag
        f = obs .- G * (mk .^ 2)
        push!(log_lines, "Iter $k: res=$res")
        
        M_diag = KernelAbstractions.ones(BACKEND, Float64, n)
        x0 = CG_GPU(G, SQS_diag, D_diag, f, M_diag, igmax, delta)
        
        m0 = copy(mk)
        dm = Q_diag .* (Sdiag .* (G' * x0))
        mk .= m0 .+ dm
        
        dres1 = obs .- G * (mk .^ 2)
        res1 = backend_dot(dres1, dres1) / n
        
        # Step adjustment
        while res1 > res
            dm ./= 3.0f0
            mk .= m0 .+ dm
            dres1 = obs .- G * (mk .^ 2)
            res1 = backend_dot(dres1, dres1) / n
            push!(log_lines, "  Step adjust: new res=$res1")
        end
        
        Sdiag .= 2 .* mk
        res = res1
        k += 1
    end
    
    # Write log file
    open("inversion.log", "w") do io
        foreach(line -> println(io, line), log_lines)
    end
    
    return Array(mk .^ 2)
end
