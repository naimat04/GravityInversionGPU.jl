# GravityInversionGPU.jl

A backend-agnostic Julia framework for 3D modeling and inversion of gravity data.

## About

This package implements a high-performance framework for three-dimensional gravity modeling and inversion in Julia. The framework addresses computational complexity, ill-posedness, and non-uniqueness in gravity inversion through:

- **Data-space inversion** to reduce dimensionality
- **Backend-agnostic implementation** using KernelAbstractions.jl
- **Multi-GPU support** (NVIDIA CUDA, Apple Metal, AMD, Intel oneAPI)
- **Advanced regularization** with depth weighting and sparsity constraints

## Quick Start

### Installation

```julia
using Pkg
Pkg.add("https://github.com/naimat04/GravityInversionGPU.jl.git")
````

Or clone and activate:

```bash
git clone https://github.com/naimat04/GravityInversionGPU.jl.git
cd GravityInversionGPU.jl
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```


### GPU Support 

This package supports multiple GPU backends. Install **only** the package for your GPU type:

| GPU Type | Package to Install |
|----------|-------------------|
| **NVIDIA** (CUDA) | `Pkg.add("CUDA")` |
| **Apple Silicon** (Metal) | `Pkg.add("Metal")` |
| **AMD** | `Pkg.add("AMDGPU")` |
| **Intel** (Arc/Xe) | `Pkg.add("oneAPI")` |

**Note**: GPU packages are optional. If none are installed, the package automatically uses CPU.


### Test the Installation

Run the included example to verify everything works:

```bash
# Test with small mesh (recommended first test)
julia --project=. examples/run_inversion.jl --nx 4 --ny 4 --nz 2

# Test with medium mesh (more realistic)
julia --project=. examples/run_inversion.jl --nx 10 --ny 10 --nz 5

# Test with parameters from paper
julia --project=. examples/run_inversion.jl --nx 40 --ny 40 --nz 20
```

After running, check the output:

```bash
ls -la examples/gravity_inversion_output_ka/
```

You should see files like `model.mesh`, `model_gpu.true`, `model_gpu.inv`, etc.

## Features

### Backend-Agnostic Computation

```julia
# Same code runs on CPU/GPU
@kernel function compute_gravity_kernel(A, cells, data)
    i, j = @index(Global, NTuple)
    # Gravity computation - runs on any backend
end
```

### Automatic GPU Detection

The framework automatically detects and uses available GPU hardware:

* **NVIDIA GPUs**: CUDA.jl backend
* **Apple Silicon**: Metal.jl backend
* **AMD GPUs**: AMDGPU.jl backend
* **Intel GPUs**: oneAPI.jl backend
* **Fallback**: CPU backend if no GPU available

### Modular Architecture

```
src/
├── GravityInversionGPU.jl          # Main module
└── modules/
    ├── CoreFunctions.jl           # Math functions
    ├── GPUBackend.jl              # GPU detection
    ├── ForwardModeling.jl         # Forward modeling
    ├── MultiGPU.jl                 # Multi-GPU sharded inversion
    ├── IOUtils.jl                 # File I/O
    └── Visualization.jl           # Plotting
```

## Performance

### GPU vs CPU Performance Comparison

|                                    | 2.0×10⁶ cells | 3.28×10⁶ cells |
| ---------------------------------- | -------------- | --------------- |
| CPU total time                     | 324.21 s       | 1007.46 s       |
| 1× A100 total time                 | 36.87 s        | 63.88 s         |
| 4× A100 total time                 | 48.57 s        | 70.27 s         |
| CPU → 1 GPU speedup                | 8.79×          | 15.77×          |
| CPU → 4 GPU speedup                | 6.67×          | 14.33×          |
| Peak GPU memory, 1× A100           | 21.85 GB       | 35.34 GB        |
| Peak GPU memory, 4× A100           | 6.21 GB/device | 10.44 GB/device |

*Note: 4-GPU sharding trades per-device memory footprint for wall-clock time — useful when a problem's memory requirement exceeds a single GPU's capacity, even though total time is somewhat higher than 1 GPU at these sizes.*

### Performance Characteristics

* **~2M cells**: single-GPU (1× A100) gives an 8.79× speedup over CPU
* **~3.28M cells**: single-GPU (1× A100) gives a 15.77× speedup over CPU
* **4-GPU sharding**: total time is somewhat higher than 1 GPU at these sizes, but peak memory per device drops substantially (e.g. 35.34 GB → 10.44 GB/device at 3.28M cells) — useful when a problem no longer fits on a single GPU

## Advanced Usage

### Custom Mesh Definition

```julia
mesh = (
    xm_min = -20.0, ym_min = -20.0, z0 = 0.0,
    dx = 500.0, dy = 500.0, dz = 500.0,
    nx = 40, ny = 40, nz = 20,
    eps = 0.1, delta = 1e-4
)

write_mesh_UBC(mesh)  # Save to UBC format
```

### Custom Inversion Parameters

```julia
# Build the (possibly multi-GPU) sharded forward operator
Gs, Qd_chunks, Dd_chunks = build_sharded_forward(mesh, xobs, yobs, delta)

# Run inversion with custom parameters
# use_squared=true enforces m = mk.^2 (positivity constraint)
inverted_model = Inversion_GPU_multi(Gs, Qd_chunks, Dd_chunks, obs_data,
                                      delta, itmax, igmax; use_squared=true)
```

### Visualization

```julia
# Generate comparison plots
composite_surface_plot(xobs, yobs, observed_data, predicted_data)
composite_model_plot(true_model, inverted_model, mesh)
```

## Output Files

The framework generates standard output files:

* `model.mesh` - Mesh definition (UBC format)
* `model_gpu.true` - True density model
* `model_gpu.inv` - Inverted model
* `data.obs` - Observed/synthetic data
* `data_gpu.pred` - Predicted data
* `data_fit_gpu.png` - Data fit visualization
* `model_plot_gpu.png` - Model comparison plots


## Examples

### Synthetic Examples

* **Two Vertical Dykes**: Tests resolution of multiple bodies

### Field Applications

* Examples from real field data are presented in our accompanying paper


## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Submit a pull request

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Contact

For questions and support:
- Open an issue on GitHub
- Nimatullah: 24D0455@iitb.ac.in
- Pankaj K. Mishra: pankaj.mishra@gtk.fi

## Acknowledgments

- Indian Institute of Technology Bombay
- Geological Survey of Finland
- Julia community for excellent tooling

---

**Note**: This is research software. Please report any issues or suggestions for improvement.
