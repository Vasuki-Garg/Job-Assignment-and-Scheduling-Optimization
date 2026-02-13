# Garment Factory Job Assignment & Scheduling (Gurobi MILP)

This repository contains a Python + Gurobi implementation of a **job assignment and scheduling** mixed-integer linear program (MILP) for a garment factory setting. The model assigns jobs to workers across discrete time periods while capturing a **learning effect** (worker productivity improves with experience), enforcing **precedence constraints**, and limiting **worker energy consumption**.

> Note: This repo provides the optimization code. You must have **Gurobi** installed and a valid license to run the solver.

## Model summary

### Decision variables (high level)
- **x(i,j,t)** ∈ {0,1}: worker *i* processes job *j* in time period *t*
- **Tmax**: makespan / maximum completion time objective
- Auxiliary variables:
  - **c(i,j,t)**: cumulative experience for worker-job pair
  - **phi(i,j,t)**: effective productivity delivered if assigned at time *t*
  - **st(j), ed(j)**: job start and end time indices
  - **zl(i,j,t,l)**: linearization binary variables for learning-productivity selection
  - **y(i,j,t)**: binary slack variables used in end-time bounding constraints (as in original formulation)

### Core constraints
- **One job per worker per period** and **one worker per job per period**
- **Learning productivity**: production rate depends on cumulative prior assignments
- **Processing requirement**: total delivered productivity must meet each job’s processing demand
- **Worker energy cap**: total energy expended per worker ≤ maximum allowed
- **Precedence constraints**: selected jobs must occur after predecessors
- **Lower bound on makespan** and optional solver cut settings

### Objective
Minimize **Tmax** (the makespan).

