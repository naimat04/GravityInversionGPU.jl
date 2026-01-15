# Forward modeling and sensitivity matrix computation

# KernelAbstractions kernel for matrix computation
@kernel function kernel_MatrixA_3D_KA!(A, Cell_grid, data, d1, d2)
    """
    GPU kernel for computing sensitivity matrix elements.
    
    Parameters:
    -----------
    A : DeviceArray
        Output sensitivity matrix
    Cell_grid : DeviceArray
        Cell center coordinates and half-thickness
    data : DeviceArray
        Observation point coordinates
    d1, d2 : Float64
        Cell dimensions in x and y directions
    """
    i, j = @index(Global, NTuple)
    
    @inbounds if i <= size(A, 1) && j <= size(A, 2)
        cx, cy, cz, ch = Cell_grid[j, 1], Cell_grid[j, 2], Cell_grid[j, 3], Cell_grid[j, 4]
        ox, oy = data[i, 1], data[i, 2]
        
        # Calculate prism corners relative to observation point
        x2 = cx + d1/2 - ox
        x1 = cx - d1/2 - ox
        y2 = cy + d2/2 - oy
        y1 = cy - d2/2 - oy
        z2 = cz + ch
        z1 = cz - ch

        # Compute gravity effect using superposition
        t1 = A_integral_single(x2, y2, z2)
        t2 = A_integral_single(x2, y2, z1)
        t3 = A_integral_single(x2, y1, z2)
        t4 = A_integral_single(x2, y1, z1)
        t5 = A_integral_single(x1, y2, z2)
        t6 = A_integral_single(x1, y2, z1)
        t7 = A_integral_single(x1, y1, z2)
        t8 = A_integral_single(x1, y1, z1)
        
        A[i, j] = t1 - t2 - t3 + t4 - t5 + t6 + t7 - t8
    end
end

function MatrixA_3D_KA_single(Cell_grid, data, d1, d2; backend=BACKEND)
    """
    Compute sensitivity matrix on GPU using KernelAbstractions.
    
    Parameters:
    -----------
    Cell_grid : Matrix{Float64}
        Cell information [x, y, z, half_thickness]
    data : Matrix{Float64}
        Observation points [x, y]
    d1, d2 : Float64
        Cell dimensions
    backend : KA.Backend
        Computational backend
        
    Returns:
    --------
    DeviceArray : Sensitivity matrix
    """
    nobs = size(data, 1)
    nCells = size(Cell_grid, 1)
    
    # Create arrays on the appropriate backend
    A = KernelAbstractions.zeros(backend, Float64, nobs, nCells)
    Cell_grid_ka = KernelAbstractions.allocate(backend, Float64, size(Cell_grid)...)
    data_ka = KernelAbstractions.allocate(backend, Float64, size(data)...)
    
    # Copy data to backend
    copyto!(Cell_grid_ka, Cell_grid)
    copyto!(data_ka, data)
    
    # Create kernel instance
    kernel! = kernel_MatrixA_3D_KA!(backend)
    
    # Configure kernel launch
    ndrange = (nobs, nCells)
    
    # Launch kernel
    ev = kernel!(A, Cell_grid_ka, data_ka, d1, d2; ndrange=ndrange)
    
    # Wait for kernel completion
    if ev !== nothing
        wait(backend, ev)
    end
    KernelAbstractions.synchronize(backend)
    
    return A
end

function Call_matrix_KA(xm_min, ym_min, xobs, yobs, z0, dx, dy, dz, nx, ny, nz, eps, delta; backend=BACKEND)
    """
    Generate forward modeling matrix with depth weighting.
    
    Parameters:
    -----------
    xm_min, ym_min, z0 : Float64
        Domain minimum coordinates
    xobs, yobs : Vector{Float64}
        Observation coordinates
    dx, dy, dz : Float64
        Cell dimensions
    nx, ny, nz : Int
        Number of cells in each dimension
    eps, delta : Float64
        Regularization parameters
    backend : KA.Backend
        Computational backend
        
    Returns:
    --------
    Tuple : (G, Q_diag, D_diag, x1, y1, z1)
    """
    m = nx * ny * nz
    n = length(xobs)
    
    # Compute cell centers
    x11 = xm_min .+ (0:nx-1) .* dx
    x1 = repeat(x11, 1, nz) |> x -> repeat(x, ny, 1) |> vec
    
    # Compute cell center coordinates in y-direction
    y11 = zeros(Float64, nx * ny)
    for i in 1:ny
        temp = ym_min + (i - 1) * dy
        for j in 1:nx
            k = (i - 1) * nx + j
            y11[k] = temp
        end
    end
    y1 = repeat(y11, nz)

    # Compute cell center coordinates in z-direction
    z1 = zeros(Float64, m)
    for i in 1:nz
        temp = (i - 1) * dz
        for j in 1:(nx * ny)
            k = (i - 1) * nx * ny + j
            z1[k] = temp
        end
    end
    
    # Build cell grid and data points
    Cell_grid = hcat(x1, y1, z1, fill(dz/2, m))
    data1 = hcat(xobs, yobs)
    
    # Compute forward matrix using KernelAbstractions
    G_ka = MatrixA_3D_KA_single(Cell_grid, data1, dx, dy; backend=backend)
    
    # Depth weighting
    q = zeros(Float64, m)
    for k in 1:nz
        wz = (z0 + (k-0.5)*dz)^3 + eps
        for j in 1:(nx*ny)
            idx = (k-1)*nx*ny + j
            q[idx] = wz
        end
    end
    Q_diag_ka = KernelAbstractions.allocate(backend, Float64, m)
    copyto!(Q_diag_ka, q)
    
    # Data covariance
    D_diag_ka = KernelAbstractions.allocate(backend, Float64, n)
    fill!(D_diag_ka, delta)
    
    return G_ka, Q_diag_ka, D_diag_ka, x1, y1, z1
end

function Gravity_response3D_GPU(model::Vector{Float64}, mesh::NamedTuple, 
                                x_obs_range::AbstractRange, y_obs_range::AbstractRange)
    """
    Generate synthetic gravity data for a given density model over a specified observation grid.
    
    Parameters
    ----------
    model : Vector{Float64}
        Density model vector of size (nx × ny × nz)
    mesh : NamedTuple
        Mesh parameters containing:
        - xm_min, ym_min, z0 : Domain minima (Float64)
        - dx, dy, dz : Cell dimensions (Float64)
        - nx, ny, nz : Number of cells in each direction (Int)
        - eps : Depth weighting parameter (Float64)
        - delta : Data covariance parameter (Float64)
    x_obs_range, y_obs_range : AbstractRange{Float64}
        Observation point ranges in x and y directions
        
    Returns
    -------
    gravity_data : Vector{Float64}
        Synthetic gravity data at observation points
    G_gpu : DeviceArray{Float64, 2}
        Forward modeling matrix on GPU
    Q_diag_gpu : DeviceArray{Float64}
        Model covariance diagonal on GPU
    D_diag_gpu : DeviceArray{Float64}
        Data covariance diagonal on GPU
    xobs, yobs : Vector{Float64}
        Observation point coordinates
    x1, y1, z1 : Vector{Float64}
        Cell center coordinates
    t : Float64
        Time taken for matrix computation
        
    Examples
    --------
    ```julia
    # Define observation grid
    x_range = 0:100:10000  # 0 to 10km with 100m spacing
    y_range = 0:100:10000
    
    # Generate synthetic data
    gravity_data, G, Q, D, xobs, yobs = 
        Gravity_response3D_GPU(model, mesh, x_range, y_range)
    ```
    """
    
    # Create observation grid
    grid = meshgrid(collect(x_obs_range), collect(y_obs_range))
    xobs, yobs = vec(grid.x), vec(grid.y)
    
    println("Observation grid: $(length(xobs)) points")
    println("X range: $(minimum(x_obs_range)) to $(maximum(x_obs_range)) m")
    println("Y range: $(minimum(y_obs_range)) to $(maximum(y_obs_range)) m")
    
    # Compute forward modeling matrix with timing
    t = @elapsed begin
        G_gpu, Q_diag_gpu, D_diag_gpu, x1, y1, z1 = Call_matrix_KA(
            mesh.xm_min, mesh.ym_min, xobs, yobs, mesh.z0,
            mesh.dx, mesh.dy, mesh.dz, mesh.nx, mesh.ny, mesh.nz,
            mesh.eps, mesh.delta
        )
    end
    
    println("Time for matrix computation: $(round(t, digits=3)) seconds")
    println("Matrix size: $(size(G_gpu)) (observations × cells)")
    
    # Transfer model to GPU
    model_ka = KernelAbstractions.allocate(BACKEND, Float64, length(model))
    copyto!(model_ka, model)
    
    # Compute gravity response: g = G * m
    gravity_data_vec = similar(model_ka, size(G_gpu, 1))
    mul!(gravity_data_vec, G_gpu, model_ka)
    gravity_data = Array(gravity_data_vec)  # Copy back to CPU
    
    # Write data to UBC format file
    output_dir = "gravity_inversion_output_ka"
    mkpath(output_dir)
    data_file = joinpath(output_dir, "synthetic_data.obs")
    
    open(data_file, "w") do io
        println(io, length(xobs))
        for i in eachindex(xobs)
            @printf(io, "%.1f %.1f 0 %.6e %.6e\n", 
                    xobs[i], yobs[i], gravity_data[i], mesh.delta)
        end
    end
    
    println("Synthetic data written to $data_file")
    println("Gravity range: [$(minimum(gravity_data)), $(maximum(gravity_data))] mGal")
    
    return gravity_data, G_gpu, Q_diag_gpu, D_diag_gpu, xobs, yobs, x1, y1, z1, t
end