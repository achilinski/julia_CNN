using JLD2
using NPZ

println("Loading Julia data...")
data_dir = joinpath(@__DIR__, "data")
prepared_data_path = joinpath(data_dir, "imdb_dataset_prepared.jld2")

X_train_julia = load(prepared_data_path, "X_train") 
y_train_julia = load(prepared_data_path, "y_train") 
X_test_julia = load(prepared_data_path, "X_test")   
y_test_julia = load(prepared_data_path, "y_test")     
embeddings_julia = load(prepared_data_path, "embeddings") 
vocab_julia = load(prepared_data_path, "vocab")

println("X_train shape: $(size(X_train_julia)), y_train shape: $(size(y_train_julia))")
println("X_test shape: $(size(X_test_julia)), y_test shape: $(size(y_test_julia))")
println("Embeddings shape: $(size(embeddings_julia))")


X_train_julia_0_based = X_train_julia .- 1
X_test_julia_0_based = X_test_julia .- 1 

println("Min/Max X_train 0-based: $(minimum(X_train_julia_0_based))/$(maximum(X_train_julia_0_based))")
println("Min/Max X_test 0-based: $(minimum(X_test_julia_0_based))/$(maximum(X_test_julia_0_based))")


X_train_pytorch = permutedims(X_train_julia_0_based, (2,1))
y_train_pytorch = permutedims(Float32.(y_train_julia), (2,1))
npzwrite("X_train_pytorch.npy", Int64.(X_train_pytorch))
npzwrite("y_train_pytorch.npy", y_train_pytorch)


X_test_pytorch = permutedims(X_test_julia_0_based, (2,1)) 
y_test_pytorch = permutedims(Float32.(y_test_julia), (2,1))   
npzwrite("X_test_pytorch.npy", Int64.(X_test_pytorch))     
npzwrite("y_test_pytorch.npy", y_test_pytorch)       


embeddings_pytorch = permutedims(Float32.(embeddings_julia), (2,1))
npzwrite("embeddings_pytorch.npy", embeddings_pytorch)

println("Data saved for PyTorch (train and test).")

julia_pad_idx_val = findfirst(x->x=="<pad>", vocab_julia)
if julia_pad_idx_val !== nothing
    pytorch_pad_idx = julia_pad_idx_val - 1
    println("Corresponding PyTorch padding_idx to use (0-based): $pytorch_pad_idx")
else
    println("WARNING: <pad> token not found in Julia vocab.")
end