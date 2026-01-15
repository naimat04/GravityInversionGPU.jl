# Input/Output utilities

function write_model_UBC(filename, model)
    """
    Write model to UBC format file.
    """
    output_dir = "gravity_inversion_output_ka"
    mkpath(output_dir)  # Create directory if it doesn't exist
    
    path = joinpath(output_dir, filename)
    open(path, "w") do io
        println(io, length(model))
        for val in model
            println(io, val)
        end
    end
    println("Model saved to $filename")
end

function write_data_UBC(filename, xobs, yobs, data, error)
    """
    Write data to UBC format file.
    """
    output_dir = "gravity_inversion_output_ka"
    mkpath(output_dir)  # Create directory if it doesn't exist
    
    path = joinpath(output_dir, filename)
    open(path, "w") do io
        println(io, length(xobs))
        for i in eachindex(xobs)
            println(io, xobs[i], " ", yobs[i], " 0 ", data[i], " ", error)
        end
    end
    println("Data saved to $filename")
end

function write_mesh_UBC(mesh)
    """
    Write mesh to UBC format file.
    """
    output_dir = "gravity_inversion_output_ka"
    mkpath(output_dir)  # Create directory if it doesn't exist
    
    path = joinpath(output_dir, "model.mesh")
    open(path, "w") do io
        println(io, mesh.nx, " ", mesh.ny, " ", mesh.nz)
        println(io, mesh.xm_min, " ", mesh.ym_min, " ", mesh.z0)
        println(io, join(fill(mesh.dx, mesh.nx), " "))
        println(io, join(fill(mesh.dy, mesh.ny), " "))
        println(io, join(fill(mesh.dz, mesh.nz), " "))
    end
    println("Mesh saved to model.mesh")
end