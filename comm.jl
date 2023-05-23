using MAT
using Distributed
@everywhere using LinearAlgebra, SparseArrays, MathProgBase, Clp, Ipopt
using JuMP
using GLPK

# Code taken from https://github.com/xjiajiahao/decg

const e  = exp(1);

# load data set, which has been partitioned to multiple nodes
function load_movie_partitioned_data(num_agents, size="1M")
    ROOT = "./data/";
    # file = matopen("data/Movies20M.mat");
    filename = "$(ROOT)Movies_$(size)_$(num_agents)_agents.mat";
    file = matopen(filename);
    user_ratings_cell_arr = read(file, "user_ratings_cell_arr"); # @NOTE we would use the cell arrary data structure where for each user, the ratings are sorted from high to low
    user_ratings_matrix = read(file, "user_ratings_matrix");
    num_movies = round(Int, read(file, "num_movies"));
    num_users = round(Int, read(file, "num_users"));
    close(file);

    return (user_ratings_cell_arr, user_ratings_matrix, num_movies, num_users)
end

function load_movie_partitioned_data_online(num_iter, size="1M")
    ROOT = "./data/";
    # file = matopen("data/Movies20M.mat");
    filename = "$(ROOT)Movies_$(size)_$(num_iter)_online_agents.mat";
    file = matopen(filename);
    user_ratings_cell_arr = read(file, "user_ratings_cell_arr"); # @NOTE we would use the cell arrary data structure where for each user, the ratings are sorted from high to low
    user_ratings_matrix = read(file, "user_ratings_matrix");
    num_movies = round(Int, read(file, "num_movies"));
    num_users = round(Int, read(file, "num_users"));
    close(file);

    return (user_ratings_cell_arr, user_ratings_matrix, num_movies, num_users)
end

# load data set, which has been randomly and equally partitioned
function load_jester_partitioned_data(num_agents, version)
    ROOT = "./data/";
    # file = matopen("data/Movies20M.mat");
    filename = "$(ROOT)Jester$(version)_$(num_agents)_agents.mat";
    file = matopen(filename);
    user_ratings_cell_arr = read(file, "user_ratings_cell_arr"); # @NOTE we would use the cell arrary data structure where for each user, the ratings are sorted from high to low
    user_ratings_matrix = read(file, "user_ratings_matrix");
    num_movies = round(Int, read(file, "num_movies"));
    num_users = round(Int, read(file, "num_users"));
    close(file);

    return (user_ratings_cell_arr, user_ratings_matrix, num_movies, num_users)
end

function load_nqp_partitioned_data(num_agents)
    ROOT = "./data/";
    filename = "$(ROOT)NQP_$(num_agents)_agents.mat";
    file = matopen(filename);
    data_cell = read(file, "data_cell"); # data_cell is a 1-by-num_agents cell, each element of data_cell is a 1-by-batch_size cell which contains #batch_size matrices H_i of size dim-by-dim
    A = read(file, "A");
    dim = round(Int, read(file, "dim"));
    u = read(file, "u");
    u = dropdims(u; dims=2);
    b = read(file, "b");
    b = dropdims(b; dims=2);
    close(file);
    return (data_cell, A, dim, u, b)
end

function load_network(network_type="er", num_agents=50)
    ROOT = "./data/";
    filename = "$(ROOT)weights_$(network_type)_$(num_agents).mat";
    file = matopen(filename);
    weights = read(file, "weights");
    close(file);
    # find the first and second largest (in magnitude) eigenvalues
    dim = size(weights, 1);
    eigvalues = (LinearAlgebra.eigen(weights)).values;
    if abs(eigvalues[dim] - 1.0) > 1e-8
        error("the largest eigenvalue of the weight matrix is $(eigvalues[dim]), but it must be 1");
    end
    beta = max(abs(eigvalues[1]), abs(eigvalues[dim - 1]));
    if beta < 1e-8
        beta = 0.0;
    end
    return (weights, beta);
end

# Linear Maximization Oracle (LMO):
# find min c^T x, s.t. a^T x < k, 0 <= x <= d, where x \in R^n, A is a m-by-n matrix, where m denotes the number of constraints
#function generate_linear_prog_function(d::Vector{Float64}, A::Array{Float64, 2}, b)
#    function linear_prog(x0) # linear programming
#        #sol = linprog(-x0, A, '<', b, 0.0, d, ClpSolver());
        #if sol.status == :Optimal 
        #    return sol.sol;
        #end
#        model = Model();
#        set_optimizer(model, GLPK.Optimizer);
#        @variable(model, 0 <= x[i=1:length(d)] <= d[i]);
#        @constraint(model, A * x .<= b);
#        @objective(model, Min, -x0'x);
#        optimize!(model);
#        if termination_status(model)==MOI.OPTIMAL
#            return value.(x);
#        else
#            error("No solution was found.");
#        end
#    end
#    return linear_prog;
#end
#
# find max <c, x>, s.t. sum(x) <= cardinality, 0 <= x <= 1
function generate_linear_prog_function(d::Vector{Float64}, cardinality::Int64)
    function linear_prog(x0) # linear programming
        dim = length(x0);
        ret = spzeros(dim);
        max_indices = partialsortperm(x0, 1:cardinality, rev=true);
        ret[max_indices] = ones(cardinality);
        return ret;
    end
    return linear_prog;
end

# Projection Oracle (PO):
# find min || x - x_0 ||, s.t. A x < b, 0 <= x <= d, where x \in R^n, a is a m-by-n matrix and m denotes the number of constraints
function generate_projection_function(d::Vector{Float64}, A::Array{Float64, 2}, b)
    n = size(A)[2]
    function projection(x0) # quadratic programming: min_x ||x - x_0 ||/2
        sol = quadprog(-x0, sparse(1.0I, n, n), A, '<', b, 0.0, d, IpoptSolver(print_level=0))
        if sol.status == :Optimal
            return sol.sol
        end 
        error("No solution was found.")
    end
    return projection
end
