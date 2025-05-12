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

using JLD2, Random, Printf, Statistics, LinearAlgebra
using Flux
using Optimisers: Adam as OptimisersAdam, setup as optimisers_setup, update! as optimisers_update!
using Profile
using StatProfilerHTML

println("External packages loaded.")

println("Loading prepared dataset...")
base_path = "."
data_path_prefix = joinpath(base_path, "data")

X_train_loaded = load(joinpath(data_path_prefix, "imdb_dataset_prepared.jld2"), "X_train")
y_train_loaded = load(joinpath(data_path_prefix, "imdb_dataset_prepared.jld2"), "y_train")
embeddings_matrix = load(joinpath(data_path_prefix, "imdb_dataset_prepared.jld2"), "embeddings")
vocab = load(joinpath(data_path_prefix, "imdb_dataset_prepared.jld2"), "vocab")
println("Dataset loaded.")

g_embedding_dim = size(embeddings_matrix, 1)
g_vocab_size = length(vocab)
g_max_len = size(X_train_loaded, 1)

g_kernel_width = 3
g_conv_out_channels = 8
g_pool_size = 8
_conv_out_width = g_max_len - g_kernel_width + 1
_pool_out_width = _conv_out_width ÷ g_pool_size
g_dense_in_features = _pool_out_width * g_conv_out_channels
g_dense_out_features = 1

g_learning_rate = 0.001f0
g_batch_size = 64
g_num_benchmark_batches = 100
g_num_samples_train = size(X_train_loaded, 2)

embeddings_for_custom_layer = permutedims(embeddings_matrix, (2,1))

padding_idx_val = findfirst(x->x=="<pad>", vocab)
if padding_idx_val === nothing; error("<pad> token not found in vocab."); end

if g_dense_in_features != 128
    @warn """Dense input feature mismatch based on calculated max_len.
             Calculated: $(( (g_max_len - g_kernel_width + 1) ÷ g_pool_size) * g_conv_out_channels),
             Notebook Dense: 128.
             Ensure max_len from data ($g_max_len) is consistent with notebook's implicit max_len for Dense(128,...).
             If max_len is 130, then calculation is correct."""
end

function get_data_for_custom(batch_idx_local)
    start_idx = (batch_idx_local-1) * g_batch_size + 1
    end_idx = min(batch_idx_local * g_batch_size, g_num_samples_train)
    actual_indices = start_idx:end_idx
    if isempty(actual_indices) || last(actual_indices) > g_num_samples_train return nothing, nothing end
    X_b_raw = X_train_loaded[:, actual_indices]
    y_b_raw = y_train_loaded[:, actual_indices]
    X_b_custom = permutedims(X_b_raw, (2,1))
    y_b_custom = permutedims(convert(Matrix{Float32}, y_b_raw), (2,1))
    return X_b_custom, y_b_custom
end

function get_data_for_flux(batch_idx_local)
    start_idx = (batch_idx_local-1) * g_batch_size + 1
    end_idx = min(batch_idx_local * g_batch_size, g_num_samples_train)
    actual_indices = start_idx:end_idx
    if isempty(actual_indices) || last(actual_indices) > g_num_samples_train return nothing, nothing end
    X_b_raw = X_train_loaded[:, actual_indices]
    y_b_raw = y_train_loaded[:, actual_indices]
    return X_b_raw, convert(Matrix{Float32}, y_b_raw)
end

println("\n--- Setting up Custom CNN ---")
cust_embedding = SimpleNN.EmbeddingLayer(g_vocab_size, g_embedding_dim, pad_idx=padding_idx_val)
SimpleNN.load_embeddings!(cust_embedding, embeddings_for_custom_layer)
cust_permute = SimpleNN.PermuteLayer((2,3,1))
cust_conv = CNNLayers.Conv1DLayer(g_kernel_width, g_embedding_dim, g_conv_out_channels, SimpleNN.relu)
cust_maxpool = CNNLayers.MaxPool1DLayer(g_pool_size, stride=g_pool_size)
cust_flatten = SimpleNN.FlattenToFeaturesBatch()
cust_transpose = SimpleNN.TransposeLayer()
cust_dense = SimpleNN.Dense(g_dense_in_features, g_dense_out_features, SimpleNN.sigmoid)
custom_model = SimpleNN.MLPModel(
    cust_embedding, cust_permute, cust_conv, cust_maxpool,
    cust_flatten, cust_transpose, cust_dense
)
custom_params = SimpleNN.get_params(custom_model)
custom_opt_state = Optimizers.Adam(g_learning_rate, custom_params)

function train_step_custom!(model_cs, optimizer_state_cs, params_cs, x_batch_indices_cs, y_true_batch_cs)
    SimpleAutoDiff.zero_grad!(params_cs)
    x_input_var_cs = SimpleAutoDiff.Variable(x_batch_indices_cs)
    y_pred_var_cs = model_cs(x_input_var_cs)
    loss_var_cs = LossFunctions.binary_cross_entropy(y_pred_var_cs, y_true_batch_cs)
    SimpleAutoDiff.backward!(loss_var_cs)
    Optimizers.update!(optimizer_state_cs)
    return SimpleAutoDiff.value(loss_var_cs)
end

println("Warm-up for Custom CNN...")
SimpleNN.train_mode!(custom_model)
xc_warm, yc_warm = get_data_for_custom(1)
if xc_warm === nothing error("Not enough data for custom warm-up batch (need at least $(g_batch_size) samples)."); end
train_step_custom!(custom_model, custom_opt_state, custom_params, xc_warm, yc_warm)
println("Custom CNN Warm-up finished.")

println("Benchmarking Custom CNN step ($(g_num_benchmark_batches) batches) WITH PROFILING...")
custom_total_time = 0.0
custom_total_allocs = 0.0
custom_batch_count = 0
SimpleNN.train_mode!(custom_model)

Profile.clear()
@profile begin
    for i in 1:g_num_benchmark_batches
        xc_b, yc_b = get_data_for_custom(i)
        if xc_b === nothing; break; end
        stats = @timed train_step_custom!(custom_model, custom_opt_state, custom_params, xc_b, yc_b)
        global custom_total_time += stats.time
        global custom_total_allocs += stats.bytes
        global custom_batch_count += 1
        if i % 50 == 0 || i == g_num_benchmark_batches
            @printf "  Custom (Profiling Batch %d/%d): Current Avg Time: %.6f s\n" i g_num_benchmark_batches (custom_total_time/custom_batch_count)
        end
    end
end

if custom_batch_count > 0
    avg_time_custom = custom_total_time / custom_batch_count
    avg_allocs_custom = custom_total_allocs / custom_batch_count
    @printf "Custom CNN (Profiled Run): Avg Time/Batch: %.6f s, Avg Allocations/Batch: %.2f MB\n" avg_time_custom (avg_allocs_custom / (1024*1024))
else
    println("Custom CNN (Profiled Run): No batches processed.")
end

println("Attempting to save flamegraph with StatProfilerHTML using defaults...")
try
    data_prof, lidict_prof = Profile.retrieve()
    if isempty(data_prof) || isempty(lidict_prof)
        println("WARNING: No profile data collected. Skipping StatProfilerHTML.")
    else
        println("Profile data collected ($(length(data_prof)) samples). Calling statprofilehtml() with defaults...")
        statprofilehtml()
        println("StatProfilerHTML default call completed. Check for 'statprofile.html'.")
    end
catch e
    println("An error occurred calling default statprofilehtml(): $e")
    Base.show_backtrace(stdout, catch_backtrace())
end

println("\n--- Setting up Flux CNN ---")
flux_embedding_layer = Flux.Embedding(g_vocab_size => g_embedding_dim)
if size(flux_embedding_layer.weight) == size(embeddings_matrix)
    flux_embedding_layer.weight .= embeddings_matrix
    println("Pre-trained embeddings loaded into Flux.Embedding layer.")
else
    error("Flux embedding weight dimension mismatch. Expected $(size(flux_embedding_layer.weight)), got $(size(embeddings_matrix)) for pre-trained.")
end

flux_model = Flux.Chain(
    flux_embedding_layer,
    x -> permutedims(x, (2,1,3)),
    Flux.Conv((g_kernel_width,), g_embedding_dim => g_conv_out_channels, Flux.relu; pad=0),
    Flux.MaxPool((g_pool_size,)),
    Flux.flatten,
    Flux.Dense(g_dense_in_features, g_dense_out_features, Flux.sigmoid)
)
flux_opt_setup = optimisers_setup(OptimisersAdam(g_learning_rate), flux_model)
loss_flux(m, x, y) = Flux.Losses.binarycrossentropy(m(x), y)

function train_step_flux!(model_fx, opt_setup_fx, x_batch_fx, y_batch_fx)
    local current_loss_fx
    params_to_grad = Flux.params(model_fx)

    current_loss_fx, grads_fx = Flux.withgradient(params_to_grad) do
        y_pred = model_fx(x_batch_fx)
        return Flux.Losses.binarycrossentropy(y_pred, y_batch_fx)
    end
    
    optimisers_update!(opt_setup_fx, params_to_grad, grads_fx) 
    
    return current_loss_fx
end

println("Warm-up for Flux CNN...")
xf_warm, yf_warm = get_data_for_flux(1)
if xf_warm === nothing error("Not enough data for Flux warm-up batch."); end
train_step_flux!(flux_model, flux_opt_setup, xf_warm, yf_warm)
println("Flux CNN Warm-up finished.")

println("Benchmarking Flux CNN step ($(g_num_benchmark_batches) batches)...")
flux_total_time = 0.0
flux_total_allocs = 0.0
flux_batch_count = 0
for i in 1:g_num_benchmark_batches
    xf_b, yf_b = get_data_for_flux(i)
    if xf_b === nothing; break; end
    stats = @timed train_step_flux!(flux_model, flux_opt_setup, xf_b, yf_b)
    global flux_total_time += stats.time
    global flux_total_allocs += stats.bytes
    global flux_batch_count += 1
    if i % 50 == 0 || i == g_num_benchmark_batches
         @printf "  Flux (Batch %d/%d): Current Avg Time: %.6f s\n" i g_num_benchmark_batches (flux_total_time/flux_batch_count)
    end
end
if flux_batch_count > 0
    avg_time_flux = flux_total_time / flux_batch_count
    avg_allocs_flux = flux_total_allocs / flux_batch_count
    @printf "Flux CNN: Avg Time/Batch: %.6f s, Avg Allocations/Batch: %.2f MB\n" avg_time_flux (avg_allocs_flux / (1024*1024))
else
    println("Flux CNN: No batches processed.")
end

println("\n--- Comprehensive Benchmarking Finished ---")
exit()