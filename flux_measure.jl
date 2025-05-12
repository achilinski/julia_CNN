using JLD2
using Flux
using Printf, Statistics
println("Starting measurement of custom_cnn_train.jl...")
println("="^50)

stats = @timed include("flux_ccn_reference.jl") 

println("="^50)
println("Finished measurement of train_cnn.jl.")
println("Total time: $(stats.time) seconds")
println("Total allocations: $(stats.bytes / (1024*1024)) MB")
println("Garbage collection time: $(stats.gctime) seconds")
println("="^50)