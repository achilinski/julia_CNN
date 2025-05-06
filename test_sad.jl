# test_sad.jl
println("Attempting to include SimpleAutoDiff.jl...")
try
    include("SimpleAutoDiff.jl")
    println("SimpleAutoDiff.jl included successfully.")
    # After include, the module is in Main
    if isdefined(Main, :SimpleAutoDiff)
        println("Module SimpleAutoDiff is defined in Main.")
        # Try using it
        # using .SimpleAutoDiff # This would be if SimpleAutoDiff was a submodule of test_sad's module
        # For a simple script, SimpleAutoDiff module is now Main.SimpleAutoDiff
        println("Trying to access SimpleAutoDiff.Variable (will error if not defined properly):")
        println(Main.SimpleAutoDiff.Variable) # Accessing it via Main
    else
        println("ERROR: Module SimpleAutoDiff is NOT defined in Main after include.")
    end
catch e
    println("ERROR during include of SimpleAutoDiff.jl:")
    showerror(stdout, e)
    Base.show_backtrace(stdout, catch_backtrace())
    println()
end