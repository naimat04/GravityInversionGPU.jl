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
    ├── Inversion.jl               # Inversion algorithms
    ├── IOUtils.jl                 # File I/O
    └── Visualization.jl           # Plotting
```

## Performance

### GPU vs CPU Performance Comparison

| Total Cells (nx×ny×nz) | CPU Time (s) | GPU Time (s) | Speedup |
| ---------------------- | ------------ | ------------ | ------- |
| 1,000                  | 0.22         | 19.25        | 0.01×   |
| 36,000                 | 0.23         | 19.23        | 0.01×   |
| 133,100                | 0.77         | 20.46        | 0.04×   |
| 1,056,000              | 0.65         | 19.94        | 0.03×   |
| 1,000,800              | 374.01       | 20.42        | 18.3×   |
| 1,458,000              | 512.19       | 20.62        | 24.8×   |
| 2,448,000              | 892.93       | 20.63        | 43.3×   |
| 3,168,000              | 1139.62      | 22.17        | 51.4×   |
| 3,213,000              | 1153.07      | 22.20        | 51.9×   |

*Note: GPU shows significant speedup for problems >1 million cells*

### Performance Characteristics

* **Small problems (<100k cells)**: CPU performs better due to GPU overhead
* **Medium problems (100k-1M cells)**: GPU begins to show advantage
* **Large problems (>1M cells)**: GPU provides 20-50× speedup

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
# Run inversion with custom parameters
inverted_model = Inversion_GPU(G_matrix, Q_diag, D_diag, obs_data,
                               delta=1e-4, itmax=10, igmax=20)
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
- Contact: 24D0455@iitb.ac.in

## Acknowledgments

- Indian Institute of Technology Bombay
- Geological Survey of Finland
- Julia community for excellent tooling

---

**Note**: This is research software. Please report any issues or suggestions for improvement.
