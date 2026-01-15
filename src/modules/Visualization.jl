# Visualization functions
using Plots

function composite_surface_plot(xobs, yobs, d_obs, inverted_data)
    """
    Create composite plot comparing observed, predicted, and residual data.
    """
    residual = d_obs .- inverted_data
    
    # Create the plots
    p1 = scatter(xobs, yobs, marker_z=d_obs,
                 xlabel="X coordinate", ylabel="Y coordinate",
                 title="Observed Gravity",
                 colormap=:viridis, colorbar=true, 
                 colorbar_title="Gravity (mGal)", legend=false,
                 markersize=5, markerstrokewidth=0)
    
    p2 = scatter(xobs, yobs, marker_z=inverted_data,
                 xlabel="X coordinate", ylabel="Y coordinate",
                 title="Predicted Gravity",
                 colormap=:viridis, colorbar=true, 
                 colorbar_title="Gravity (mGal)", legend=false,
                 markersize=5, markerstrokewidth=0)
    
    p3 = scatter(xobs, yobs, marker_z=residual,
                 xlabel="X coordinate", ylabel="Y coordinate",
                 title="Residual (Obs - Pred)",
                 colormap=:viridis, colorbar=true, 
                 colorbar_title="Difference (mGal)", legend=false,
                 markersize=5, markerstrokewidth=0)
    
    # Combine plots
    composite = plot(p1, p2, p3, layout=(1,3), size=(1200,400))
    
    # Save to file
    output_dir = "gravity_inversion_output_ka"
    mkpath(output_dir)
    save_path = joinpath(output_dir, "data_fit_gpu.png")
    savefig(composite, save_path)
    println("✓ Plot saved to $save_path")
    
    return composite
end

function composite_model_plot(model_true, model_rec, mesh)
    """
    Create composite plot comparing true and recovered models.
    """
    nx, ny, nz = mesh.nx, mesh.ny, mesh.nz
    
    # Reshape to 3D
    true3d = reshape(model_true, (nx, ny, nz))
    rec3d = reshape(model_rec, (nx, ny, nz)) 
    resid3d = true3d .- rec3d
    
    # Take a middle slice
    slice_index = max(1, Int(round(nz / 2)))
    
    # Create the plots
    p1 = heatmap(true3d[:,:,slice_index]', aspect_ratio=:equal,
                 title="True Model (Slice z=$slice_index)", 
                 xlabel="X index", ylabel="Y index", 
                 colorbar=false, colorbar_title="Density",
                 clims=(minimum(model_true), maximum(model_true)))
    
    p2 = heatmap(rec3d[:,:,slice_index]', aspect_ratio=:equal,
                 title="Recovered Model (Slice z=$slice_index)", 
                 xlabel="X index", ylabel="Y index", 
                 colorbar=false, colorbar_title="Density",
                 clims=(minimum(model_true), maximum(model_true)))
    
    p3 = heatmap(resid3d[:,:,slice_index]', aspect_ratio=:equal,
                 title="Residual (True - Recovered)", 
                 xlabel="X index", ylabel="Y index", 
                 colorbar=false, colorbar_title="Density Diff")
    
    # Combine plots
    composite = plot(p1, p2, p3, layout=(1,3), size=(1200,400))
    
    # Save to file
    output_dir = "gravity_inversion_output_ka"
    mkpath(output_dir)
    save_path = joinpath(output_dir, "model_plot_gpu.png")
    savefig(composite, save_path)
    println("✓ Plot saved to $save_path")
    
    return composite
end