using Test, SparseArrays
using Recommenders: σ, Logloss, ElasticNet, grad, cd!

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

@testset "ElasticNet" begin
    elasticnet = ElasticNet(0.2, 0.1)
    @test elasticnet(rand(2, 2), zeros(2), zeros(2))==0
    @test elasticnet(sprand(2, 2, 0.5), spzeros(2), spzeros(2))==0
    X = [1 0; 1 1]
    y = [1, 0]
    w = [1, 1]
    @test elasticnet(X, y, w) == 1.0 + 0.2*0.1*2 + 0.5*0.2*0.9*2

    elasticnet = ElasticNet(0., 0.)
    y=[1.5]
    x = 2.0 .* ones(1,1)
    w = [0.2]
    cd!(elasticnet, x, y, w)
    @test w == [0.75]
    w=[-3.0]
    cd!(elasticnet, x, y, w)
    @test w == [0.75]

    elasticnet = ElasticNet(0.5, 0.1)
    y=[1.,0.]
    X=[1. 0.;-1. 2.]
    w=[0.2, -0.4]
    cd!(elasticnet, X, y, w)
    # I derived the numbers below by the elementary calculation by hand.
    @test w ≈ [0.1/2.9, -0.0]
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
