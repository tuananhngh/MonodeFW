using Distributed
@everywhere using Random

# Code taken from https://github.com/xjiajiahao/decg

@everywhere function f_extension(x, ratings) # ratings is a n-by-2 matrix sorted in descendant order, where n denotes #movies some user has rated
    res = 0;
    prod = 1;
    for index in 1:size(ratings, 2) # for all rated movies from high to low
        x_current_coord = x[round(Int, ratings[1, index])];
        res += ratings[2, index] * x_current_coord * prod;
        prod *= 1 - x_current_coord;
        if prod == 0
            break;
        end
    end
    return res;
end

@everywhere function f_extension_batch(x, batch_ratings) # ratings is a n-by-2 matrix sorted in descendant order, where n denotes #movies some user has rated
    sum_f = 0;
    batch_size = length(batch_ratings);
    #display(batch_size;)
    for ratings in batch_ratings
        sum_f += f_extension(x, ratings);
    end
    return sum_f/batch_size;
end

@everywhere function partial_extension(x, ratings, i) # compute partial derivative : O(#ratings)
    res_union = 0;
    res_diff = 0;
    prod = 1;

    # index_of_i_in_ratings = findfirst(ratings[1, :], i);  # @NOTE bottleneck
    index_of_i_in_ratings = 0;  # the for loop is equivilant to the above line, but faster
    for tmp_i in 1 : size(ratings, 2)
        if i == ratings[1, tmp_i]
            index_of_i_in_ratings = tmp_i;
            break;
        end
    end

    if index_of_i_in_ratings == 0 # this means f(R+{i}) = f(R\{i}), for any R, then no need to sample
        return 0;
    end

    for index in 1:size(ratings, 2) # for all rated movies from high to low
        if index == index_of_i_in_ratings
            res_union = res_diff + ratings[2, index] * 1 * prod;
        else
            x_current_coord = x[round(Int, ratings[1, index])];
            res_diff += ratings[2, index] * x_current_coord * prod;  # @NOTE bottleneck
            prod *= 1 - x_current_coord;
        end
        if prod == 0
            break;
        end
    end
    return res_union - res_diff;
end

@everywhere function gradient_extension(x, ratings) # compute the gradient: O(n + #sample * #ratings^2)
    dim = length(x);
    res = zeros(dim);
    for i in 1:dim
        res[i] = partial_extension(x, ratings, i);
    end
    return res;
end

@everywhere function gradient_extension_batch(x, batch_ratings) # ratings is a n-by-2 matrix sorted in descendant order, where n denotes #movies some user has rated
    dim = length(x);
    sum_gradient = zeros(dim);
    for ratings in batch_ratings
        sum_gradient += gradient_extension(x, ratings);
    end
    return sum_gradient;
end


@everywhere function stochastic_gradient_extension!(x::Vector{Float64}, ratings::Array{Float64, 2}, 
        sample_times::Int64, indices_in_ratings::Vector{Int64}, 
        ret_stochastic_grad::Vector{Float64}, rand_vec::Vector{Float64})
    dim = length(x);
    fill!(indices_in_ratings, zero(Int));
    tmp_idx = 0;
    nnz = size(ratings, 2);
    for i = 1 : nnz
        tmp_idx = round(Int, ratings[1, i]);
        indices_in_ratings[i] = tmp_idx;
    end

    rand_vec_view = view(rand_vec, 1:nnz);
    for j = 1:sample_times
        Random.rand!(rand_vec_view);

        max_1st_rating_in_S = 0; max_1st_index_in_rating = 0; max_1st_index_in_x = 0;
        max_2nd_rating_in_S = 0;
        count = 0;
        # find the first and second largest rating in S
        for i = 1 : nnz
            tmp_index = indices_in_ratings[i];
            if rand_vec_view[i] <= x[tmp_index]
                if count == 0
                    max_1st_rating_in_S = ratings[2, i];
                    max_1st_index_in_rating = i;
                    max_1st_index_in_x = tmp_index;
                    ret_stochastic_grad[max_1st_index_in_x] += max_1st_rating_in_S;
                    count += 1;
                else
                    max_2nd_rating_in_S = ratings[2, i];
                    ret_stochastic_grad[max_1st_index_in_x] -= max_2nd_rating_in_S;
                    break;
                end
            end
            if count == 0
                ret_stochastic_grad[tmp_index] += ratings[2, i];
            end
        end

        for i = 1 : max_1st_index_in_rating - 1
            tmp_index = indices_in_ratings[i];
            ret_stochastic_grad[tmp_index] -= max_1st_rating_in_S;
        end
    end
    nothing
end

@everywhere function stochastic_gradient_extension_batch(x, batch_ratings, sample_times = 1) # ratings is a n-by-2 matrix sorted in descendant order, where n denotes #movies some user has rated
    dim = length(x);
    stochastic_gradient = zeros(dim);
    indices_in_ratings = zeros(Int64, dim);
    rand_vec = zeros(dim);
    for ratings in batch_ratings
        stochastic_gradient_extension!(x, ratings, sample_times, indices_in_ratings, stochastic_gradient, rand_vec);
    end
    stochastic_gradient = stochastic_gradient ./ sample_times;
    return stochastic_gradient;
end

