# src/MyCNNProjectModule.jl
module MyCNNProjectModule

# Include your components. The `.` makes the path relative to this file.
include("SimpleAutoDiff.jl")
include("SimpleNN.jl")
include("CNN.jl")
include("Loss.jl")
include("Optimizer.jl")

# Optionally re-export names if you want to use MyCNNProjectModule.MyType
# Or make them submodules and export from them
# For example, to make SimpleAutoDiff directly usable after `using MyCNNProjectModule`:
using .SimpleAutoDiff # Make SimpleAutoDiff's exports available
# You might want to be more explicit or create submodules within MyCNNProjectModule

end