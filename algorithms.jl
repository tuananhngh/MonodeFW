
mutable struct FPL
    x
    eta
    accumulated_gradient
    n0
end;

function monode_fw(dim, data_cell, num_agents, weights, num_out_edges, LMO, f_batch, gradient_batch, num_iters)
    usr_by_agent = round(Int,size(data_cell[1],2)/num_agents);
    usr_by_agent_idx = [(i-1)*usr_by_agent + 1:i*usr_by_agent for i in 1:num_agents];
    function gradient_cat(x, data)
        grad_x = @sync @distributed (hcat) for i in 1:num_agents
            gradient_batch(x[:,i], data[usr_by_agent_idx[i]],10)
        end
        return grad_x;
    end
    function f_sum(x, data)
        f_x = @sync @distributed (+) for i in 1:num_agents
            f_batch(x, data[usr_by_agent_idx[i]])
        end
        return f_x;
    end
    function LMO_cat(d)
        res = @sync @distributed (hcat) for i in 1:num_agents
            LMO(d[:, i])
        end
        return res;
    end
    function get_vector(fpl_)
        return fpl_.x
    end
    
    function update(fpl_, gradient)
        fpl_.accumulated_gradient += gradient;
        sol = LMO_cat(fpl_.accumulated_gradient*fpl_.eta .+ fpl_.n0);
        fpl_.x = sol
    end
    
    Random.seed!(1234);
    K = floor(Int,num_iters^(3/5)); 
    Q = round(Int,num_iters^(2/5));
    
    n0 = randn(dim, num_agents);
    x0 = LMO_cat(n0);
    fpl = [FPL(x0, 1/(K*Q)^(1/2), zeros(dim, num_agents), n0) for k in 1:K];
    
    perm = [randperm(K) for q=1:Q];
    #display(perm)
    num_comm = 0.0;
    reward = zeros(K*Q);
    t_start = time();
    xq = zeros(dim, num_agents)
    index = 1;
    for q in 1:Q
        #println("Q : $(q)");
        xs = zeros(dim, K+1, num_agents);
        gs_ = zeros(dim, K, num_agents);
        as_ = zeros(dim, num_agents);
        perm_grad = zeros(dim, K, num_agents);
        v = [get_vector(fpl[k]) for k in 1:K];
        
        for k in 1:K
            ys_ = xs[:,k,:]*weights;
            #xs[:,k+1,:] = xs[:,k,:] + v[k]/K;
            xs[:,k+1,:] = ys_ .+ v[k]/K;
        end
        xq = xs[:,K+1,:];
        for k in 1:K
            index = k+K*(q-1);
            sgma = indexin(k, perm[q])[1];
            perm_grad[:,k,:] = gradient_cat(xs[:,sgma,:], data_cell[index]);
            tmp_rw = zeros(num_agents);
            for i in 1:num_agents
                tmp_rw[i] = f_sum(xq[:,i],data_cell[index])/num_agents;
            end
            #println("Index $(index) $(tmp_rw)");
            reward[index] = minimum(tmp_rw);
        end
        gs_[:,1,:] = perm_grad[:,perm[q][1],:];
        for k in 1:K
            if k < K/2 +1
                rho = min(2*(k+3)^(-2/3),1);
            else
                rho = min(1.5*(K-k+2)^(-2/3),1);
            end
            ds = gs_[:,k,:]*weights;
            num_comm += dim*num_out_edges;
            as_ = (1-rho)*as_ + rho*ds;
            if k < K
                gs_[:,k+1,:] = perm_grad[:, perm[q][k+1],:] - perm_grad[:,perm[q][k],:] + ds;
            end
            update(fpl[k], as_);
        end
    end
    t_elapse = time() - t_start
    result = [K*Q, t_elapse, num_comm, xq];
    return result, reward
end;


function frank_wolfe(dim, gradient, lmo, K)
    x = zeros(dim)
    for k in 1:K
        grad_x = gradient(x)
        v = lmo(grad_x)
        x += v / K
    end
    return x
end

function offline(dim, data, f, gradient, lmo, K)
    T = size(data,2);
    reward = zeros(T);
    gradient_cumul = x->sum([gradient(x, params) for params in data]);
    v = frank_wolfe(dim, gradient_cumul, lmo, K);
    for i in 1:T
        reward[i] = f(v,data[i]);
    end
    return reward;
end