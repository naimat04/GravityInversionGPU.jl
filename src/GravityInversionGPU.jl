module GravityInversionGPU

# Import required packages
import LinearAlgebra, SparseArrays, Plots, Printf, ArgParse, KernelAbstractions

# Create and export output directory
const output_dir = "gravity_inversion_output_ka"
export output_dir

# Include all modules (they define functions in the global scope)
include("modules/CoreFunctions.jl")
include("modules/GPUBackend.jl")
include("modules/IOUtils.jl")  # This defines functions, not a module
include("modules/ForwardModeling.jl")
include("modules/Inversion.jl")
include("modules/Visualization.jl")

# Re-export everything users need
export 
    # Constants
    output_dir, BACKEND,
    
    # From CoreFunctions
    my_formatter, A_integral_single, meshgrid, backend_dot,
    
    # From IOUtils
    write_model_UBC, write_data_UBC, write_mesh_UBC,
    
    # From ForwardModeling
    Gravity_response3D_GPU, Call_matrix_KA, MatrixA_3D_KA_single,
    
    # From Inversion
    Inversion_GPU, CG_GPU,
    
    # From Visualization
    composite_surface_plot, composite_model_plot

end
