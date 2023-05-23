using Dates, MAT, Statistics, JLD;
include("comm.jl");
include("facility_location.jl");
include("algorithms.jl");

save_path = "./results3/";

function settings(num_agents, num_iter, graph_style, cardinality)
    data_cell, data_mat, num_movies, num_users = load_movie_partitioned_data_online(num_iter, "1M");
    available_graph_style = ["complete", "line", "er","grid"];
    if ~(graph_style in available_graph_style)
        error("graph_style should be \"complete\", \"line\", or \"er\"");
    end
    weights, beta = load_network(graph_style, num_agents);
    num_out_edges = count(i->(i>0), weights) - num_agents;
    dim = num_movies;
    x0 = zeros(dim);

    d = ones(dim);
    a_2d = ones(1, dim); # a should be a n_constraints-by-dim matrix
    LMO = generate_linear_prog_function(d, cardinality);
    return weights, dim, data_cell, LMO, num_out_edges
end;

function run_monodefw(num_agents, graph_styles, num_iter, cardinality)
    #all_graph_res = [];
    for k in cardinality
        println("-----K $(k)-----")
        for graph in graph_styles
            graph_res = [];
            println("-----$(graph) Graph-----");
            for nodes in num_agents
                println("-----#Agents $(nodes)-----");
                weights, dim, data, LMO, num_out_edges = settings(nodes, num_iter, graph, k);
                _, rw = monode_fw(dim, data, nodes, weights, num_out_edges, 
                                LMO, f_extension_batch, stochastic_gradient_extension_batch, num_iter);
                push!(graph_res, rw);
            end
            #push!(all_graph_res, graph_res);
            save(joinpath(save_path,"res_monode_k$(k)_$(graph).jld"), graph, graph_res)
        end
    end
    #return all_graph_res;
end;

function run_offline(num_iter, cardinality)
    num_agents = 1;
    num_iter = 120;
    data_cell, data_mat, num_movies, num_users = load_movie_partitioned_data_online(num_iter, "1M");
    dim = num_movies;
    #x0 = zeros(dim);
    d = ones(dim);
    #a_2d = ones(1, dim); # a should be a n_constraints-by-dim matrix
    for k in cardinality
        println("----Running $(k) Cardinality-----");
        LMO = generate_linear_prog_function(d, k);
        rw_offline = offline(dim, data_cell, f_extension_batch, gradient_extension_batch, LMO, 50);
        save(joinpath(save_path,"res_off_k$(k).jld"),"offline", rw_offline);
    end
end;

num_agents = [10, 25, 50];
graph_styles = ["complete","er","grid","line"];
num_iter = 120;
cardinality = [10,20,30,40,50];



