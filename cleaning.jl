using JLD2
using FileIO

folder = "Saved_content"
files = readdir(folder)

for file in files
    if endswith(file, ".jld2") && !occursin("copy", file)
        path = joinpath(folder, file)
        try
            vars = keys(load(path))
            if isempty(vars)
                println("Deleting empty file: $file")
                rm(path)
            end
        catch e
            @warn "Error reading $file, deleting as possibly corrupted" exception=e
            rm(path)  # Delete unreadable/corrupt file
        end
    elseif occursin("copy", file)
        println("Deleting file with 'copy' in name: $file")
        rm(joinpath(folder, file))
    end
end