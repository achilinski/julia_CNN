# MLP.jl
module MLPDefinition

using .SimpleAutoDiff
using Random, LinearAlgebra, Statistics

# Add DropoutLayer, train_mode!, eval_mode! to exports
export Dense, EmbeddingLayer, FlattenLayer, MeanPoolLayer, DropoutLayer, # New
       relu, sigmoid, tanh_approx,
       MLPModel, forward, get_params, load_embeddings!,
       train_mode!, eval_mode! # New

# --- Activation Functions --- (Unchanged)
# ... relu, sigmoid, tanh_approx ...
relu(x::Variable{T}) where T<:Real = max(x, T(0.0))
function sigmoid(x::Variable{T}; ϵ=1e-8) where T<:Real; one_val=ones(T, size(value(x))); one_var=Variable(one_val,is_param=false); exp_neg_x=exp(-x); denominator=one_var+exp_neg_x+Variable(fill(Base.eps(T), size(value(x))),is_param=false); return one_var/denominator; end
function tanh_approx(x::Variable{T}) where T<:Real; two=Variable(fill(T(2.0),size(value(x))),is_param=false); one=Variable(fill(T(1.0),size(value(x))),is_param=false); return two*SimpleAutoDiff.sigmoid(two*x)-one; end

# --- Dense Layer --- (Unchanged)
# ... Dense struct, forward, get_params ...
struct Dense; W::Variable; b::Variable; activation::Function; function Dense(i,o,a=identity;dtype=Float32); l=sqrt(dtype(6)/(dtype(i)+dtype(o))); Wv=rand(dtype,i,o).*dtype(2).*l.-l; bv=zeros(dtype,1,o); W=Variable(Wv,is_param=true); b=Variable(bv,is_param=true); new(W,b,a); end; end
function forward(layer::Dense, x::Variable{T}) where T; linear_out=matmul(x, layer.W)+layer.b; return layer.activation(linear_out); end
function get_params(layer::Dense); return [layer.W, layer.b]; end

# --- Embedding Layer --- (Unchanged)
# ... EmbeddingLayer struct, forward, get_params, load_embeddings! ...
struct EmbeddingLayer; weight::Variable; vocab_size::Int; embedding_dim::Int; function EmbeddingLayer(vs,ed;dtype=Float32); l=sqrt(dtype(6)/(dtype(vs)+dtype(ed))); Wv=(rand(dtype,vs,ed).*dtype(2).*l.-l).+dtype(1e-4); w=Variable(Wv,is_param=true); new(w,vs,ed); end; end
function forward(layer::EmbeddingLayer, x_indices::AbstractMatrix{<:Integer}, pad_idx::Int); T=SimpleAutoDiff._eltype(layer.weight.value); bs,sl=size(x_indices); os=(bs,sl,layer.embedding_dim); ov=zeros(T,os); for b=1:bs,t=1:sl; idx=x_indices[b,t]; if idx!=pad_idx && 1<=idx<=layer.vocab_size; ov[b,t,:].=view(layer.weight.value,idx,:); end; end; children=Variable[layer.weight]; local nv; function bf(); ogv=grad(nv); ign=ogv===nothing ? 0f0 : norm(ogv); if rand()<0.001; println("\n [Emb Bwd] In grad norm: ",round(ign,sigdigits=4)); end; if ogv===nothing; @warn "..."; SimpleAutoDiff.accumulate_gradient!(layer.weight,zeros(T,size(value(layer.weight)))); return; end; Tl=SimpleAutoDiff._eltype(layer.weight.value); wga=zeros(Tl,size(value(layer.weight))); bsl,sll=size(x_indices); npi=false; for b=1:bsl,t=1:sll; idx=x_indices[b,t]; if idx!=pad_idx && 1<=idx<=layer.vocab_size; npi=true; if b<=size(ogv,1) && t<=size(ogv,2) && idx<=size(wga,1); gs=view(ogv,b,t,:); wgr=view(wga,idx,:); wgr.+=gs; end; end; end; cwgn=norm(wga); if rand()<0.001 || (cwgn==0f0 && npi); println(" [Emb Bwd] Calc W grad norm: ", round(cwgn, sigdigits=4)); end; SimpleAutoDiff.accumulate_gradient!(layer.weight,wga); fsgn=layer.weight.gradient===nothing ? 0f0 : norm(layer.weight.gradient); if rand()<0.001 || (fsgn==0f0 && npi); println(" [Emb Bwd] Final stored W grad norm: ", round(fsgn, sigdigits=4)); end; end; nv=Variable(ov,children,bf); return nv; end
function get_params(layer::EmbeddingLayer); return [layer.weight]; end
function load_embeddings!(layer::EmbeddingLayer, embeddings_matrix::AbstractMatrix); if size(embeddings_matrix,2)!=layer.embedding_dim; println("Err dim"); return; end; if size(embeddings_matrix,1)!=layer.vocab_size; println("Err vocab"); return; end; layer.weight.value.=embeddings_matrix; println("Embed loaded."); end

# --- Flatten Layer --- (Unchanged)
# ... FlattenLayer struct, forward, get_params ...
struct FlattenLayer; input_shape::Ref{Tuple}; FlattenLayer()=new(Ref{Tuple}(())); end
function forward(layer::FlattenLayer, x::Variable{T}) where T; iv=value(x); isv=size(iv); bs=isv[1]; nf=prod(isv[2:end]); os=(bs,nf); layer.input_shape[]=isv; ov=reshape(iv,os); children=Variable[x]; local nv; function bf(); ogv=grad(nv); if ogv!==nothing; igv=reshape(ogv,layer.input_shape[]); SimpleAutoDiff.accumulate_gradient!(x,igv); end; end; nv=Variable(ov,children,bf); return nv; end
get_params(layer::FlattenLayer)=[];

# --- MeanPoolLayer --- (Unchanged)
# ... MeanPoolLayer struct, forward, get_params ...
struct MeanPoolLayer; dims; MeanPoolLayer(dims)=new(dims); end
function forward(layer::MeanPoolLayer, x::Variable{T}) where T; return Statistics.mean(x; dims=layer.dims); end
get_params(layer::MeanPoolLayer)=[];

# --- NEW: Dropout Layer ---
mutable struct DropoutLayer{T<:Real}
    p::T # Dropout probability (probability of ZEROING an element)
    is_training::Ref{Bool} # Use RefValue to allow modification
    # Store mask used in forward pass for use in backward pass
    # Use RefValue for mask to update it inplace in forward
    mask::Ref{Union{Nothing, AbstractArray{T}}}

    function DropoutLayer(p::T; is_training_ref::Base.RefValue{Bool}=Ref(true)) where {T<:Real}
        if !(0 <= p < 1); error("Dropout probability p must be in [0, 1)"); end
        new{T}(p, is_training_ref, Ref{Union{Nothing, AbstractArray{T}}}(nothing)) # Initialize mask Ref as nothing
    end
end

function forward(layer::DropoutLayer{T}, x::Variable{T}) where {T<:Real}
    # If not training, just return the input Variable
    if !layer.is_training[]
        return x
    end

    # --- Training Mode ---
    val = value(x)
    p = layer.p

    # Generate mask (true means KEEP, false means DROP)
    # Use the same type as the value for the mask elements (often Bool, but could be Float32)
    # mask_bool = rand(eltype(val), size(val)) .> p # Original thought - Bool mask
    mask_val = rand!(similar(val)) .> p # Inplace rand & compare -> Bool mask
    layer.mask[] = convert(AbstractArray{T}, mask_val) # Store mask as T (0.0 or 1.0)

    # Apply mask and scale (Inverted Dropout)
    scale_factor = T(1.0 / (1.0 - p))
    output_val = (val .* layer.mask[]) .* scale_factor

    # Define backward pass
    children = Variable[x]
    local new_var
    function backward_fn()
        output_grad = grad(new_var)
        # Gradient exists AND mask was actually generated (should be true if training)
        if output_grad !== nothing && layer.mask[] !== nothing
            # Apply the same mask and scaling used during forward pass
            input_grad = (output_grad .* layer.mask[]) .* scale_factor
            accumulate_gradient!(x, input_grad)
        # else # Handle case where grad is nothing, maybe accumulate zero?
             # accumulate_gradient!(x, zero(value(x)))
        end
         # Clear the mask after backward pass? Not strictly necessary if overwritten next forward pass
         # layer.mask[] = nothing
    end

    new_var = Variable(output_val, children, backward_fn)
    return new_var
end

# Dropout layer has no trainable parameters
get_params(layer::DropoutLayer) = []


# --- MLP Model (Chain Abstraction) ---
struct MLPModel
    layers::Vector{Any}
    parameters::Vector{Variable}
    # Share training mode across dropout layers
    is_training_ref::Ref{Bool}

    function MLPModel(layers...)
        model_layers = [l for l in layers]
        params = Variable[]
        is_training_ref = Ref(true) # Default to training mode

        actual_layers = []
        for layer in model_layers
            # If layer is Dropout, potentially link its is_training ref
            # Simplest: Assume user passes pre-configured Dropout layers or handle externally
            push!(actual_layers, layer)
            if hasmethod(get_params, (typeof(layer),))
                append!(params, get_params(layer))
            end
        end
        new(actual_layers, params, is_training_ref)
    end
end

# --- Helper functions to set training/evaluation mode ---
function train_mode!(model::MLPModel)
    # println("Setting model to TRAINING mode")
    model.is_training_ref[] = true
    # Propagate to layers that care (like Dropout)
    for layer in model.layers
        if layer isa DropoutLayer
            layer.is_training[] = true
        end
    end
end

function eval_mode!(model::MLPModel)
    # println("Setting model to EVALUATION mode")
    model.is_training_ref[] = false
    for layer in model.layers
        if layer isa DropoutLayer
            layer.is_training[] = false
        end
    end
end


# Default forward - assumes simple chain.
# Modify IF layers need shared state like training mode passed explicitly
# function forward(model::MLPModel, x::Variable{T}) where T ...
# (model::MLPModel)(x::Variable{T}) where T = forward(model, x)
# Let's keep using manual forward calls in main.jl for now.

function get_params(model::MLPModel)
    return model.parameters
end


end # module MLPDefinition