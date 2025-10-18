# MonodeFW

Companion source code for the paper:

**One Gradient Frank-Wolfe for Decentralized Online Convex and Submodular Optimization**
TA Nguyen, NK Thang, D Trystram
*Asian Conference on Machine Learning (ACML), 2023*

[Paper PDF](https://proceedings.mlr.press/v189/nguyen23a/nguyen23a.pdf)

## Overview

This repository implements the **MonODe-FW** (Monotone Online Decentralized Frank-Wolfe) algorithm for solving decentralized online optimization problems over networks. The algorithm is designed for monotone submodular and convex optimization in a distributed setting where multiple agents collaborate over a network topology.

## Repository Structure

- **[algorithms.jl](algorithms.jl)** - Core implementation of the MonODe-FW algorithm
- **[comm.jl](comm.jl)** - Network communication utilities and graph topology setup (complete, line, Erdős-Rényi, grid graphs)
- **[facility_location.jl](facility_location.jl)** - Facility location problem formulation using movie recommendation as a submodular optimization example
- **[main_experiment_facility_location.jl](main_experiment_facility_location.jl)** - Main experimental script for running facility location experiments
- **[plot_regret.jl](plot_regret.jl)** - Visualization tools for plotting regret curves
- **[data/](data/)** - MovieLens datasets (100K and 1M) partitioned for multiple agents, and network weight matrices for various graph topologies

## Problem Setting

The code addresses decentralized online optimization where:
- Multiple agents connected via a network topology
- Each agent receives data sequentially over time
- Agents communicate only with their neighbors
- Goal: Minimize cumulative regret for monotone submodular/convex functions
- Constraint: Linear matroid constraints solved via Linear Maximization Oracle (LMO)

## Experiments

The main experiments use the **MovieLens** dataset (100K and 1M ratings) to solve a facility location problem where agents collaboratively recommend movies to maximize user satisfaction. The code evaluates performance across different:
- Network topologies (complete, line, Erdős-Rényi, grid)
- Number of agents (10, 25, 50, 60, 100, 120)
- Cardinality constraints
- Online iterations

## Requirements

- Julia (tested with distributed computing support)
- Required packages: `Distributed`, `MAT`, `Statistics`, `JLD`, `LinearAlgebra`, `SparseArrays`, `JuMP`, `GLPK`, `Clp`, `Ipopt`

## Usage

```julia
# Load required modules
include("comm.jl")
include("facility_location.jl")
include("algorithms.jl")

# Run experiments with specific parameters
run_monodefw(num_agents, graph_styles, num_iter, cardinality)
```

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{nguyen2023one,
  title={One Gradient Frank-Wolfe for Decentralized Online Convex and Submodular Optimization},
  author={Nguyen, TA and Thang, NK and Trystram, D},
  booktitle={Asian Conference on Machine Learning},
  pages={989--1004},
  year={2023},
  organization={PMLR}
}
```
