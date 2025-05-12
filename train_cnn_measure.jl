println("Including local modules...")
include("SimpleAutoDiff.jl") 
include("SimpleNN.jl")       
include("CNNLayers.jl")            
include("LossFunctions.jl")
include("Optimizers.jl")
println("Local modules included.")



using .SimpleAutoDiff
using .SimpleNN
using .CNNLayers      
using .LossFunctions  
using .Optimizers      
using ProfileView
using Profile
using StatProfilerHTML
using PProf
using JLD2, Random, Printf, Statistics, LinearAlgebra, InteractiveUtils 
println("Starting measurement of custom_cnn_train.jl...")
println("="^50)

stats = @timed include("train_cnn.jl") 

println("="^50)
println("Finished measurement of train_cnn.jl.")
println("Total time: $(stats.time) seconds")
println("Total allocations: $(stats.bytes / (1024*1024)) MB")
println("Garbage collection time: $(stats.gctime) seconds")
println("="^50)