# Core mathematical functions and constants
using Printf

# Custom tick formatter for plots
function my_formatter(x)
    x == 0 && return "0"
    exp = round(Int, log10(abs(x)))
    abs(x - 10.0^exp) < 1e-6 ? "10^$(exp)" : @sprintf("%.0f", x)
end

# GPU-compatible A_integral function
function A_integral_single(x, y, z)
    """
    Compute the analytical solution for the gravity integral of a single prism.
    
    Parameters:
    -----------
    x, y, z : Float64
        Coordinates relative to prism center
        
    Returns:
    --------
    Float64 : Gravity contribution
    """
    Gamma = 6.674e-3  # Gravitational constant in mGal·m²/kg
    T = promote_type(eltype(x), eltype(y), eltype(z))
    r = sqrt(x^2 + y^2 + z^2)
    
    # Handle division by zero in atan
    denom = z * r
    atan_arg = (abs(denom) > 1e-12) ? (x * y) / denom : T(0)
    
    # Analytical formula for prism gravity effect
    f = -Gamma * (x * log(y + r) + y * log(x + r) - z * atan(atan_arg))
    return f
end

# GPU-compatible meshgrid function
function meshgrid(xin, yin)
    """
    Create 2D grid coordinates from 1D vectors.
    
    Parameters:
    -----------
    xin, yin : Vector{Float64}
        Input coordinate vectors
        
    Returns:
    --------
    NamedTuple : (x, y) matrices
    """
    nx, ny = length(xin), length(yin)
    xout = [xin[j] for i in 1:ny, j in 1:nx]
    yout = [yin[i] for i in 1:ny, j in 1:nx]
    return (x = xout, y = yout)
end

# Helper function for dot product that works on all backends
function backend_dot(a, b)
    """
    Compute dot product compatible with various GPU backends.
    
    Parameters:
    -----------
    a, b : AbstractArray
        Input vectors
        
    Returns:
    --------
    Float64 : Dot product result
    """
    return sum(a .* b)
end