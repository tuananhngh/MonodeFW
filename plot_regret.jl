using JLD;
using Plots;
using Statistics;

alpha = 1-1/exp(1);
path_result = "./results3/";
list_result_files = readdir(path_result);

save_path = "./img3/";

k_20 = [res for res in list_result_files if occursin("k20", res)];


complete_monode = load(joinpath(path_result,k_20[1]))["complete"];
er_monode = load(joinpath(path_result,k_20[2]))["er"];
grid_monode = load(joinpath(path_result,k_20[3]))["grid"];
line_monode = load(joinpath(path_result,k_20[4]))["line"];
#offline_val = load(joinpath(path_result,k_10[4]))["offline"];


function regret(graph_rw, off_rw)
    ratios = [];
    regrets = [];
    for result in graph_rw
        ratio = result./off_rw[1:119];
        rw = alpha*cumsum(off_rw[1:119]) .- cumsum(result);
        push!(ratios, ratio);
        push!(regrets, rw);
    end
    return ratios, regrets
end;

function cumsum_value(graph_rw)
    vals = [];
    for result in graph_rw
        val = mean(result);
        push!(vals, val);
    end;
    return vals;
end;

#complete_ratio, complete_regret = regret(complete_monode, offline_val);
#line_ratio, line_regret

function ratio_figure(ratio_val, graph_type, k)
    nb_ratio = size(ratio_val)[1];
    p = plot(ratio_val[1], label="$(graph_type)-10",legend=:topleft);
    for (i, node) in zip(2:nb_ratio, [25,50])
        plot!(ratio_val[i], label="$(graph_type)-$(node)");
    end;
    xlabel!("Iteration Index")
    ylabel!("Ratio")
    p;
    savefig(p, joinpath(save_path,"k$(k)/ratio_$(graph_type).png"));
end;


function regret_figure(regret_val, graph_type,k)
    nb_ratio = size(regret_val)[1];
    p = plot(regret_val[1], label="$(graph_type)-10",legend=:topleft);
    for (i, node) in zip(2:nb_ratio, [25,50])
        plot!(regret_val[i], label="$(graph_type)-$(node)");
    end;
    xlabel!("Iteration Index")
    ylabel!("(1-1/e)-Regret")
    p;
    savefig(p,joinpath(save_path,"k$(k)/regret_$(graph_type).png"));
end;

function figure_by_k(k, path, res_files)
    k_res = [res for res in res_files if occursin("k$(k)", res)];
    comp_graph = load(joinpath(path,k_res[1]))["complete"];
    er_graph = load(joinpath(path,k_res[2]))["er"];
    grid_graph = load(joinpath(path,k_res[3]))["grid"];
    line_graph = load(joinpath(path,k_res[4]))["line"];
    offline = load(joinpath(path,k_res[5]))["offline"];

    ratio_k, regret_k = regret(comp_graph, offline); 
    ratio_er, regret_er = regret(er_graph, offline);
    ratio_grid, regret_grid = regret(grid_graph, offline);
    ratio_line, regret_line = regret(line_graph, offline);

    regrets = [regret_k, regret_er, regret_grid, regret_line];
    ratios = [ratio_k, ratio_er, ratio_grid, ratio_line];
    g_styles = ["K", "ER","Grid","Line"];
    for (rat,reg, g) in zip(ratios, regrets, g_styles)
        display(regret_figure(reg, g, k));
        display(ratio_figure(rat, g, k));
    end;
end;

function get_all_k(graph_style, nb_agent, path, res_files)
    g_res = [res for res in res_files if occursin("$(graph_style)", res)];
    graph_k_vals = [];
    for k_val in g_res
        comp_i = load(joinpath(path,k_val))["$(graph_style)"][nb_agent];
        sum_comp = mean(comp_i);
        push!(graph_k_vals, sum_comp);
    end;
    return graph_k_vals;
end;

function figure_all_k(path, nb_agent, res_files)
    list_graphs = ["er","grid","line"];
    list_k = [10,20,30,40,50];
    complete_val = get_all_k("complete",nb_agent, path, res_files);
    #g_offline = [res for res in res_files if occursin("off", res)];
    #offline_val = [];
    #for k_val in g_offline
    #    tmp = mean(load(joinpath(path,k_val))["offline"]);
    #    push!(offline_val, tmp);
    #end;
    p = plot(list_k,complete_val, label="K",legend=:topleft, markershape=:auto);
    for g in list_graphs
        tmp = get_all_k(g, nb_agent, path, res_files);
        plot!(list_k,tmp, label=uppercase("$(g)"), markershape=:auto);
    end
    #plot!(list_k, (1-1/exp(1)).*offline_val, label="FW-Offline", markershape=:auto);
    xlabel!("k - Cardinality")
    ylabel!("Average Objective Value")
    p;
    savefig(p,joinpath(save_path,"avg_$(nb_agent).png"));
end;

function compare_graph_by_k(k, path, nb_agents, res_files)
    list_graph = ["complete", "er", "grid", "line"];
    k_res = [res for res in res_files if occursin("k$(k)", res)];
    comp_graph = load(joinpath(path,k_res[1]))["complete"];
    er_graph = load(joinpath(path,k_res[2]))["er"];
    grid_graph = load(joinpath(path,k_res[3]))["grid"];
    line_graph = load(joinpath(path,k_res[4]))["line"];
    offline = load(joinpath(path,k_res[5]))["offline"];

    _, regret_k = regret(comp_graph, offline); 
    _, regret_er = regret(er_graph, offline);
    _, regret_grid = regret(grid_graph, offline);
    _, regret_line = regret(line_graph, offline);

    regrets = [regret_er[nb_agents], regret_grid[nb_agents], regret_line[nb_agents]];
    g_styles = ["ER","Grid","Line"];
    p = plot(regret_k[nb_agents], label="K",legend=:topleft);
    for (reg, g) in zip(regrets, g_styles)
        plot!(reg, label=uppercase("$(g)"));
    end;
    xlabel!("Iteration Index")
    ylabel!("(1-1/e)-Regret")
    p;
end;