# Optimizer.jl
module Optimizers

using ..SimpleAutoDiff
using Statistics # For sqrt

export SGD, Adam, RMSProp, update!

# --- SGD Optimizer ---
mutable struct SGD{T<:Real}
    lr::T
    params::Vector{<:SimpleAutoDiff.Variable{T}}
    function SGD(lr::T, params::Vector{<:SimpleAutoDiff.Variable{T}}) where {T<:Real}
        trainable_params = filter(p -> p.is_param, params)
        if length(trainable_params) != length(params); @warn "SGD initialized with non-parameter Variables."; end
        new{T}(lr, trainable_params)
    end
end

function update!(opt::SGD{T}) where {T<:Real}
    for p in opt.params
        if p.gradient !== nothing
            grad_val = grad(p)
# --- START CHANGE (SGD Check) ---
            # Check for non-finite values (NaN or Inf) correctly for scalars and arrays
            is_non_finite = (isa(grad_val, Real) && !isfinite(grad_val)) ||
                            (isa(grad_val, AbstractArray) && !all(isfinite, grad_val))
            if is_non_finite
                 @warn "SGD: Non-finite gradient detected for parameter shape $(size(p.value)). Skipping update."
                 continue
            end
# --- END CHANGE (SGD Check) ---
            p.value .-= opt.lr .* grad_val
        end
    end
end


# --- Adam Optimizer ---
mutable struct Adam{T<:Real}
    lr::T
    beta1::T
    beta2::T
    epsilon::T
    params::Vector{<:SimpleAutoDiff.Variable{T}}
    m::Dict{SimpleAutoDiff.Variable{T}, Union{T, AbstractArray{T}}}
    v::Dict{SimpleAutoDiff.Variable{T}, Union{T, AbstractArray{T}}}
    t::Int
    function Adam(lr::T, params::Vector{<:SimpleAutoDiff.Variable}; beta1::Real=0.9, beta2::Real=0.999, epsilon::Real=1e-8) where {T<:Real}
        trainable_params = Vector{SimpleAutoDiff.Variable{T}}()
        for p in params; if p.is_param && p isa SimpleAutoDiff.Variable{T}; push!(trainable_params, p); end; end
        if isempty(trainable_params); @warn "Adam initialized with no trainable parameters of type $T."; end
        m_dict = Dict{SimpleAutoDiff.Variable{T}, Union{T, AbstractArray{T}}}()
        v_dict = Dict{SimpleAutoDiff.Variable{T}, Union{T, AbstractArray{T}}}()
        for p in trainable_params; m_dict[p] = zero(p.value); v_dict[p] = zero(p.value); end
        new{T}(T(lr), T(beta1), T(beta2), T(epsilon), trainable_params, m_dict, v_dict, 0)
    end
end

function update!(opt::Adam{T}) where {T<:Real}
    opt.t += 1
    bias_correction1 = one(T) - opt.beta1^opt.t + T(1e-9)
    bias_correction2 = one(T) - opt.beta2^opt.t + T(1e-9)
    for p in opt.params
        if p.gradient !== nothing
            g = grad(p)
# --- START CHANGE (Adam Check) ---
            is_non_finite = (isa(g, Real) && !isfinite(g)) ||
                            (isa(g, AbstractArray) && !all(isfinite, g))
            if is_non_finite
                 @warn "Adam: Non-finite gradient detected for parameter shape $(size(p.value)). Skipping update."
                 continue
            end
# --- END CHANGE (Adam Check) ---
            opt.m[p] .= opt.beta1 .* opt.m[p] .+ (one(T) - opt.beta1) .* g
            opt.v[p] .= opt.beta2 .* opt.v[p] .+ (one(T) - opt.beta2) .* (g .^ 2)
            m_hat = opt.m[p] ./ bias_correction1
            v_hat = opt.v[p] ./ bias_correction2
            update_step = opt.lr .* m_hat ./ (sqrt.(v_hat) .+ opt.epsilon)
            p.value .-= update_step
        end
    end
end


# --- RMSProp Optimizer ---
mutable struct RMSProp{T<:Real}
    lr::T
    rho::T
    epsilon::T
    params::Vector{<:SimpleAutoDiff.Variable{T}}
    accumulators::Dict{SimpleAutoDiff.Variable{T}, Union{T, AbstractArray{T}}}
    function RMSProp(lr::T, params::Vector{<:SimpleAutoDiff.Variable}; rho::Real=0.9, epsilon::Real=1e-6) where {T<:Real}
        trainable_params = Vector{SimpleAutoDiff.Variable{T}}()
        for p in params; if p.is_param && p isa SimpleAutoDiff.Variable{T}; push!(trainable_params, p); end; end
        if isempty(trainable_params); @warn "RMSProp initialized with no trainable parameters of type $T."; end
        acc_dict = Dict{SimpleAutoDiff.Variable{T}, Union{T, AbstractArray{T}}}()
        for p in trainable_params; acc_dict[p] = zero(p.value); end
        new{T}(T(lr), T(rho), T(epsilon), trainable_params, acc_dict)
    end
end

function update!(opt::RMSProp{T}) where {T<:Real}
    for p in opt.params
        if haskey(opt.accumulators, p) && p.gradient !== nothing
            g = grad(p)
# --- START CHANGE (RMSProp Check) ---
            # ERROR WAS HERE: Trying to call isnan/isinf directly on array `g`
            # if (isa(g, Real) && (isnan(g) || isinf(g))) || (isa(g, AbstractArray) && (any(isnan, g) || any(isinf, g)))
            # Correct Check: Use isfinite for both scalars and arrays (via `all`)
            is_non_finite = (isa(g, Real) && !isfinite(g)) ||
                            (isa(g, AbstractArray) && !all(isfinite, g))
            if is_non_finite
                 @warn "RMSProp: Non-finite gradient detected for parameter shape $(size(p.value)). Skipping update."
                 continue
            end
# --- END CHANGE (RMSProp Check) ---
            acc = opt.accumulators[p]
            acc .= opt.rho .* acc .+ (one(T) - opt.rho) .* (g .^ 2)
            update_step = (opt.lr ./ (sqrt.(acc) .+ opt.epsilon)) .* g
            p.value .-= update_step
        end
    end
end
function clip_gradients!(params::Vector{<:Variable}, threshold::Real)
    if threshold <= 0; error("Gradient clipping threshold must be positive."); end

    # Calculate global L2 norm squared first (more numerically stable than summing norms)
    # Use Float64 for accumulation to avoid overflow on large gradients
    global_norm_sq::Float64 = 0.0
    for p in params
        if p.is_param && p.gradient !== nothing
            grad_val = p.gradient # Already the value (Matrix or Scalar)
            global_norm_sq += sum(abs2, grad_val) # Sum of squares
        end
    end
    global_norm = sqrt(global_norm_sq)

    # Calculate the clipping factor (if needed)
    clip_coef = Float32(threshold) / (Float32(global_norm) + eps(Float32)) # Use target dtype, add epsilon

    # Apply clipping only if the norm exceeds the threshold
    if clip_coef < 1.0
        # println("Clipping Gradients: Norm = $global_norm, Coef = $clip_coef") # Debug print
        for p in params
            if p.is_param && p.gradient !== nothing
                p.gradient .*= clip_coef # Scale gradient inplace
            end
        end
    end
    return global_norm # Return the original norm (optional)
end

end # module Optimizers