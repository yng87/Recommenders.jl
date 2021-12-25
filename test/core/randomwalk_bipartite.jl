using Test, DataFrames
using Recommenders:
    get_degree,
    get_max_degree,
    onewalk,
    randomwalk,
    pixie_multi_hit_boost,
    aggregate_multi_randomwalk,
    randomwalk_multiple,
    build_graph

@testset "Get degree function." begin
    offsets = [1, 2, 4, 10]
    @test get_degree(offsets, 1) == 1
    @test get_degree(offsets, 2) == 2
    @test get_degree(offsets, 3) == 6
    @test get_degree(offsets, [1, 3, 2]) == [1, 6, 2]
    @test get_max_degree(offsets) == 6
end

@testset "One walk" begin
    """
    adjacency_list
    User:
        1: [4, 5]
        2: [5]
        3: [4,5]
    Item:
        4: [1, 3]
        5: [1, 2, 3]
    """
    adjacency_list = [4, 5, 5, 4, 5, 1, 3, 1, 2, 3]
    offsets = [1, 3, 4, 6, 8, 11]

    @test onewalk(adjacency_list, offsets, 1) in [4, 5]
    @test onewalk(adjacency_list, offsets, 2) == 5
    @test onewalk(adjacency_list, offsets, 3) in [4, 5]
    @test onewalk(adjacency_list, offsets, 4) in [1, 3]
    @test onewalk(adjacency_list, offsets, 5) in [1, 2, 3]
end

@testset "One random walk." begin
    """
    adjacency_list
    User:
        1: [4, 5]
        2: [5]
        3: [4,5]
    Item:
        4: [1, 3]
        5: [1, 2, 3]
    """
    adjacency_list = [4, 5, 5, 4, 5, 1, 3, 1, 2, 3]
    offsets = [1, 3, 4, 6, 8, 11]

    # same node type
    for node in keys(randomwalk(adjacency_list, offsets, 1, true, 0.1, 100, Inf, Inf))
        @test node in [1, 2, 3]
    end

    for node in keys(randomwalk(adjacency_list, offsets, 2, true, 0.1, 100, 2, 10))
        @test node in [1, 2, 3]
    end

    for node in keys(randomwalk(adjacency_list, offsets, 4, true, 0.1, 10, 2, 10))
        @test node in [4, 5]
    end

    # different node type
    for node in keys(randomwalk(adjacency_list, offsets, 1, false, 0.1, 100, Inf, Inf))
        @test node in [4, 5]
    end

    for node in keys(randomwalk(adjacency_list, offsets, 2, false, 0.1, 100, 2, 10))
        @test node in [4, 5]
    end

    for node in keys(randomwalk(adjacency_list, offsets, 4, false, 0.1, 10, 2, 10))
        @test node in [1, 2, 3]
    end
end

@testset "Simple aggregate." begin
    visited_counts = [Dict(2 => 4, 3 => 2)]
    @test aggregate_multi_randomwalk(visited_counts) == Dict(2 => 4, 3 => 2)
    @test aggregate_multi_randomwalk(visited_counts, maximum) == Dict(2 => 4, 3 => 2)

    visited_counts = [Dict(1 => 2, 2 => 5), Dict(2 => 4, 3 => 2)]
    @test aggregate_multi_randomwalk(visited_counts) == Dict(1 => 2, 2 => 9, 3 => 2)
    @test aggregate_multi_randomwalk(visited_counts, maximum) ==
          Dict(1 => 2, 2 => 5, 3 => 2)
end

@testset "Pixie boosting." begin
    visited_counts = [Dict(1 => 2, 2 => 5)]
    total_visited_count = pixie_multi_hit_boost(visited_counts)
    @test total_visited_count[1] ≈ 2.0
    @test total_visited_count[2] ≈ 5.0

    visited_counts = [Dict(1 => 2, 2 => 5), Dict(2 => 4, 3 => 2)]
    total_visited_count = pixie_multi_hit_boost(visited_counts)
    @test total_visited_count[1] ≈ 2.0
    @test total_visited_count[2] ≈ (sqrt(5) + sqrt(4))^2
    @test total_visited_count[3] ≈ 2.0
end

@testset "Random walk multiple." begin
    """
    adjacency_list
    User:
        1: [4, 5]
        2: [5]
        3: [4,5]
    Item:
        4: [1, 3]
        5: [1, 2, 3]
    """
    adjacency_list = [4, 5, 5, 4, 5, 1, 3, 1, 2, 3]
    offsets = [1, 3, 4, 6, 8, 11]

    # same node type
    total_visited_count = randomwalk_multiple(
        adjacency_list,
        offsets,
        [4, 5],
        true,
        0.3,
        100,
        2,
        5,
        false,
        false,
    )
    for (node, c) in total_visited_count
        @test node in [4, 5]
        @test c > 0
    end

    total_visited_count = randomwalk_multiple(
        adjacency_list,
        offsets,
        [4, 5],
        true,
        0.3,
        100,
        2,
        5,
        true,
        false,
    )
    for (node, c) in total_visited_count
        @test node in [4, 5]
        @test c > 0
    end

    total_visited_count =
        randomwalk_multiple(adjacency_list, offsets, [1], true, 0.3, 100, 2, 5, false, true)
    for (node, c) in total_visited_count
        @test node in [1, 2, 3]
        @test c > 0
    end

    total_visited_count = randomwalk_multiple(
        adjacency_list,
        offsets,
        [1, 3],
        true,
        0.3,
        100,
        2,
        5,
        true,
        true,
    )
    for (node, c) in total_visited_count
        @test node in [1, 2, 3]
        @test c > 0
    end

    total_visited_count = randomwalk_multiple(
        adjacency_list,
        offsets,
        [1, 3],
        true,
        0.3,
        100,
        2,
        5,
        true,
        true,
        3,
    )
    for (node, c) in total_visited_count
        @test node in [1, 2, 3]
        @test c > 0
    end

    # different node type
    total_visited_count = randomwalk_multiple(
        adjacency_list,
        offsets,
        [4, 5],
        false,
        0.3,
        100,
        2,
        5,
        false,
        false,
    )
    for (node, c) in total_visited_count
        @test node in [1, 2, 3]
        @test c > 0
    end

    total_visited_count = randomwalk_multiple(
        adjacency_list,
        offsets,
        [1, 3],
        false,
        0.3,
        100,
        2,
        5,
        true,
        true,
        3,
    )
    for (node, c) in total_visited_count
        @test node in [4, 5]
        @test c > 0
    end
end

@testset "Bipartite graph build" begin
    df = DataFrame(itemid = [1, 1, 2, 3, 3], userid = [1, 2, 2, 1, 2])

    df_original = copy(df)

    adjacency_list, offsets, user2uidx, item2iidx = build_graph(df)
    @test adjacency_list == [4, 5, 5, 4, 5, 1, 3, 1, 2, 3]
    @test offsets == [1, 3, 4, 6, 8, 11]
    @test user2uidx == Dict(1 => 4, 2 => 5)
    @test item2iidx == Dict(1 => 1, 2 => 2, 3 => 3)

    # check if df is mutated.
    @test df == df_original
end
