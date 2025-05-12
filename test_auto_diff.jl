include("SimpleAutoDiff.jl")
using ..SimpleAutoDiff
using Test 

println("--- Running SimpleAutoDiff Correctness Tests ---")

@testset "Simple Scalar Operations" begin
    
    x1 = Variable(3.0f0)
    y1 = x1^2
    backward!(y1)
    @test grad(x1) ≈ 6.0f0 atol=1e-6

    zero_grad!(x1) 

    
    x2 = Variable(1.0f0)
    y2 = exp(x2)
    backward!(y2)
    @test grad(x2) ≈ exp(1.0f0) atol=1e-6

    zero_grad!(x2)

    x3 = Variable(0.5f0)
    y3_param = Variable(2.0f0)
  
    z3 = (x3 * y3_param) + (x3^3)
    backward!(z3)
    @test grad(x3) ≈ 2.75f0 atol=1e-6
    @test grad(y3_param) ≈ 0.5f0 atol=1e-6

    zero_grad!(x3)
    zero_grad!(y3_param)

    
    x4 = Variable(2.0f0)
    y4 = log(x4)
    backward!(y4)
    @test grad(x4) ≈ 0.5f0 atol=1e-6
end

@testset "Vector/Matrix Operations" begin
    
    
    v1_val = Float32[1.0, 2.0, 3.0]
    v1 = Variable(v1_val)
    s1 = sum(v1)
    backward!(s1)
    @test grad(v1) ≈ ones(Float32, 3) atol=1e-6

    zero_grad!(v1)
    a2_val = Float32[1.0, 2.0]
    b2_val = Float32[3.0, 4.0]
    a2 = Variable(a2_val)
    b2 = Variable(b2_val)
    y2 = sum(a2 * b2)
    backward!(y2)
    @test grad(a2) ≈ b2_val atol=1e-6
    @test grad(b2) ≈ a2_val atol=1e-6

    zero_grad!(a2)
    zero_grad!(b2)

    A_val = Float32[1.0 2.0; 3.0 4.0] 
    X_val_vec = Float32[5.0; 6.0]
    X_val_mat = reshape(X_val_vec, 2, 1)

    A = Variable(A_val, is_param=true)
    X = Variable(X_val_mat, is_param=true)

    Y_matmul = matmul(A, X) 
    L_sum = sum(Y_matmul)   
    backward!(L_sum)

    expected_grad_A_correct = ones(Float32, 2, 1) * transpose(X_val_mat)
    @test grad(A) ≈ expected_grad_A_correct atol=1e-5 

    expected_grad_X_correct = transpose(A_val) * ones(Float32, 2, 1)
    @test grad(X) ≈ expected_grad_X_correct atol=1e-5

end
println("--- SimpleAutoDiff Correctness Tests Finished ---")