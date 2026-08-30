module GravityInversionGPU

# Import required packages
import LinearAlgebra, SparseArrays, Plots, Printf, ArgParse, KernelAbstractions, CUDA

# Create and export output directory
const output_dir = "gravity_inversion_output_ka"
export output_dir

# Include all modules (they define functions in the global scope)
include("modules/CoreFunctions.jl")
include("modules/GPUBackend.jl")
include("modules/IOUtils.jl")  # This defines functions, not a module
include("modules/ForwardModeling.jl")
include("modules/Visualization.jl")
include("modules/MultiGPU.jl")  # Multi-GPU row-sharded forward/inversion

export 
    # Constants
    output_dir, BACKEND,
    
    # From CoreFunctions
    my_formatter, A_integral_single, meshgrid, backend_dot,
    
    # From IOUtils
    write_model_UBC, write_data_UBC, write_mesh_UBC,
    
    # From ForwardModeling
    Gravity_response3D_GPU, Call_matrix_KA, MatrixA_3D_KA_single,
    
    # From Visualization
    composite_surface_plot, composite_model_plot,

    # From MultiGPU
    ShardedMatrix, build_sharded_forward, Inversion_GPU_multi, CG_GPU_multi,
    predict_multi, assign_devices, partition_rows,
    replicate_to_devices, scatter_to_devices, gather_from_devices

end