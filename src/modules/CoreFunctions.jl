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
    x, y, z : Float32 or Float64
        Coordinates relative to prism center (type follows caller)

    Returns:
    --------
    Gravity contribution, same float type as x, y, z
    """
    T = promote_type(eltype(x), eltype(y), eltype(z))
    Gamma = T(6.674e-3)  # Gravitational constant in mGal·m²/kg, cast to match input type
                          # IMPORTANT: this must stay type-matched to x/y/z. If this were
                          # left as a bare Float64 literal, every GPU thread would silently
                          # promote back to Float64 mid-kernel, which (a) defeats the memory
                          # savings of a Float32 G matrix and (b) can cause GPU kernel
                          # compilation/type-instability issues since GPU kernels require
                          # type-stable code.
    r = sqrt(x^2 + y^2 + z^2)

    # Handle division by zero in atan
    denom = z * r
    atan_arg = (abs(denom) > T(1e-12)) ? (x * y) / denom : T(0)

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
    xin, yin : Vector
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
    Dot product result (same float type as inputs)
    """
    return sum(a .* b)
end