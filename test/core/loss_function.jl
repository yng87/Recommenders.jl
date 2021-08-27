using Test
using Recommenders: σ, Logloss, grad

@testset "Sigmoid." begin
    @test σ(0.0) == 0.5
    @test σ(100) ≈ 1.0
    @test σ(-100) < 1e-20
    @test σ(1.0) ≈ 1.0 / (1 + exp(-1))
    @test σ(0.1) > 0.5
    @test σ(-0.1) < 0.5
end

@testset "Logloss." begin
    logloss = Logloss()
    @test logloss(100, 1) ≈ 0
    @test logloss(-100, 1) > 20
    @test logloss(3, 0) > 3
    @test logloss(-100, 0) ≈ 0
    @test logloss(0.0, 1) == logloss(0.0, 0)
end

@testset "Grad of Logloss." begin
    logloss = Logloss()

    @test grad(logloss, 0.3243, 0) ≈ σ(0.3243)
    @test grad(logloss, 0.3243, 1) ≈ σ(0.3243) - 1

    # check sign
    logit = 0.4
    label = 1
    loss = logloss(logit, label)
    grad_logloss = grad(logloss, logit, label)
    @test logloss(logit - 0.1 * grad_logloss, label) < loss

    label = 0
    loss = logloss(logit, label)
    grad_logloss = grad(logloss, logit, label)
    @test logloss(logit - 0.1 * grad_logloss, label) < loss


    logit = -0.3
    label = 1
    loss = logloss(logit, label)
    grad_logloss = grad(logloss, logit, label)
    @test logloss(logit - 0.1 * grad_logloss, label) < loss

    label = 0
    loss = logloss(logit, label)
    grad_logloss = grad(logloss, logit, label)
    @test logloss(logit - 0.1 * grad_logloss, label) < loss
end
