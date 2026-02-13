from gurobipy import Model, GRB, quicksum
import gurobipy as gp
import math
import pandas as pd
import os
import argparse


def solve(excel_path: str, sheet_name: str = "Part 1"):
    # read information of the process
    Process_Dataset_df = pd.read_excel(excel_path, sheet_name=sheet_name)

    Volume_Job = Process_Dataset_df["Processing Time (1 min)"].tolist()
    Skill_Level = Process_Dataset_df["Required Skill Level"].tolist()

    # convert the skill level to the required energy consumption per 10 seconds (length of period)
    EnergyConsm = []
    for letter in Skill_Level:
        if letter == "A":
            EnergyConsm.append(3)
        elif letter == "B":
            EnergyConsm.append(2.8)
        elif letter == "C":
            EnergyConsm.append(2.5)
        elif letter == "D":
            EnergyConsm.append(2.3)
        else:
            raise ValueError(f"Unknown skill level: {letter}")

    # create a list of precedence (first one is job i and the second is precedence)
    Precedence = [
        [11, 1], [12, 1], [13, 2], [14, 2], [15, 3], [16, 4],
        [17, 4], [18, 5], [18, 6], [19, 7], [20, 8], [21, 8],
        [21, 9], [22, 10], [23, 24], [25, 26], [27, 24]
    ]

    # Parameters
    jWj = 8  # Number of Workers
    jJj = len(Volume_Job)  # Number of Jobs
    jTj = 8  # Number of Time Periods (each period represent i minute)
    Kij = 2  # Learning Asymptote
    pij = 1  # Initial Experience
    rij = 3  # Learning rate

    v = Volume_Job  # Processing time for the each job
    ME = 18.6  # Max work in the time horizon
    E = EnergyConsm

    # Model
    model = gp.Model("Job_Assignment_with_Learning")

    # Sets
    W = range(1, jWj + 1)
    J = range(1, jJj + 1)
    T = range(1, jTj + 1)

    # Variables Sets
    x = {}
    c = {}
    y = {}
    st = {}
    ed = {}
    phi = {}
    zl = {}
    phi_l_cap = {}
    M = {}
    Phi_l_CapT = {}
    Phi_l_bar_CapT = {}

    Tmax = model.addVar(vtype=GRB.CONTINUOUS, name="Tmax")

    # Variables Creation
    for i in W:
        for j in J:
            for t in T:
                production_rates = []
                for l in range(t):
                    phi_l_cap[i, j, t, l] = Kij * (1 - math.exp(-(l + pij) / rij))
                    production_rates.append(phi_l_cap[i, j, t, l])
                M[i, j, t] = max(production_rates)

    for i in W:
        for j in J:
            productivity = []
            for l in range(jTj):
                productivity.append(phi_l_cap[i, j, jTj, l])
            Phi_l_CapT[i, j] = max(productivity)

    for j in J:
        productivity = []
        for i in W:
            productivity.append(Phi_l_CapT[i, j])
        Phi_l_bar_CapT[j] = max(productivity)

    for i in W:
        for j in J:
            for t in T:
                x[i, j, t] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}_{t}")
                c[i, j, t] = model.addVar(vtype=GRB.CONTINUOUS, name=f"c_{i}_{j}_{t}")
                phi[i, j, t] = model.addVar(vtype=GRB.CONTINUOUS, name=f"phi_{i}_{j}_{t}")
                y[i, j, t] = model.addVar(vtype=GRB.BINARY, name=f"y_{i}_{j}_{t}")

    for i in W:
        for j in J:
            for t in T:
                for l in range(t + 1):
                    zl[i, j, t, l] = model.addVar(vtype=GRB.BINARY, name=f"zl_{i}_{j}_{t}_{l}")

    for j in J:
        st[j] = model.addVar(vtype=GRB.INTEGER, name=f"st_{j}")
        ed[j] = model.addVar(vtype=GRB.INTEGER, name=f"ed_{j}")

    # Objective function
    model.setObjective(Tmax, GRB.MINIMIZE)

    # Constraints
    # Constraint 1
    for i in W:
        for j in J:
            for t in T:
                model.addConstr(t * x[i, j, t] <= Tmax, f"constraint_1_{i}_{j}_{t}")

    # Constraint 2
    for i in W:
        for t in T:
            model.addConstr(quicksum(x[i, j, t] for j in J) <= 1, f"constraint_2_{i}_{t}")

    # Constraint 3
    for j in J:
        for t in T:
            model.addConstr(quicksum(x[i, j, t] for i in W) <= 1, f"constraint_3_{j}_{t}")

    # Constraint 4
    for i in W:
        for j in J:
            for t in range(2, jTj + 1):
                model.addConstr(
                    c[i, j, t] == quicksum(x[i, j, k] for k in range(1, t)),
                    f"constraint_4_{i}_{j}_{t}"
                )

    # Constraint 5
    for i in W:
        for j in J:
            for t in T:
                model.addConstr(
                    phi[i, j, t] == quicksum(phi_l_cap[i, j, t, l] * zl[i, j, t, l] for l in range(0, t)),
                    f"constraint_5_{i}_{j}_{t}"
                )

    # Constraint 6
    for i in W:
        for j in J:
            for t in T:
                model.addConstr(
                    quicksum(l * zl[i, j, t, l] for l in range(0, t)) <= c[i, j, t],
                    f"constraint_6_{i}_{j}_{t}"
                )

    # Constraint 7
    for i in W:
        for j in J:
            for t in T:
                model.addConstr(
                    quicksum(zl[i, j, t, l] for l in range(0, t + 1)) <= 1,
                    f"constraint_7_{i}_{j}_{t}"
                )

    # Constraint 8
    for i in W:
        for j in J:
            for t in T:
                model.addConstr(
                    phi[i, j, t] <= M[i, j, t] * x[i, j, t],
                    f"constraint_8_{i}_{j}_{t}"
                )

    # Constraint 9
    for j in J:
        model.addConstr(
            quicksum(phi[i, j, t] for i in W for t in T) >= v[j - 1],
            f"constraint_9_{j}"
        )

    # Constraint 10
    for i in W:
        model.addConstr(
            quicksum(E[j - 1] * x[i, j, t] for j in J for t in T) <= ME,
            f"constraint_10_{i}"
        )

    # Constraint 11: V.I
    for j in J:
        model.addConstr(
            quicksum(x[i, j, t] for i in W for t in T) >= math.ceil(v[j - 1] / Phi_l_bar_CapT[j]),
            f"constraint_11_{j}"
        )

    # Constraint 12: Jobs Precedence
    for combination in Precedence:
        model.addConstr(
            st[combination[0]] - ed[combination[1]] >= 1,
            f"constraint_12_{combination}"
        )

    # Constraint 12-1
    for j in J:
        for i in W:
            for t in T:
                model.addConstr(
                    ed[j] <= t * x[i, j, t] + jTj * y[i, j, t],
                    f"constraint_14_{i}_{j}_{t}"
                )

    # Constraint 12-3
    for j in J:
        model.addConstr(
            quicksum(y[i, j, t] for i in W for t in T) == jWj * jTj - 1,
            f"constraint_15"
        )

    # Constraint 12-5
    for i in W:
        for t in T:
            for j in J:
                model.addConstr(ed[j] >= t * x[i, j, t], f"constraint_16{j}")

    # Constraint 12-6
    for j in J:
        model.addConstr(ed[j] <= Tmax, f"constraint_17")

    # Constraint 12-7
    for j in J:
        model.addConstr(
            ed[j] - st[j] == quicksum(x[i, j, t] for i in W for t in T) - 1,
            f"constraint_18_{j}"
        )

    # Constraint 12-8
    for j in J:
        model.addConstr(st[j] >= 1, f"constraint_19_{j}")

    # Constraint 12-9
    for j in J:
        model.addConstr(
            st[j] <= quicksum(t * x[i, j, t] for i in W for t in T),
            f"constraint_20_{j}"
        )

    # Constraint 13: Lower Bound
    model.addConstr(Tmax >= 5)

    # Implement Cuts Created by Gurobi
    model.setParam('Cuts', 3)  # Enable Gurobi to generate Chvátal-Gomory cuts

    # Solving model
    model.optimize()

    # Print the values of decision variable x
    if model.status == GRB.OPTIMAL:
        print("Objective value with Chvátal-Gomory cuts:", model.objVal)
        print("Solution with Chvátal-Gomory cuts:")
        for var in model.getVars():
            if var.varName.startswith("x") and var.x == 1:
                print(
                    f"Worker {var.varName.split('_')[1]} performs job {var.varName.split('_')[2]} "
                    f"at period {var.varName.split('_')[3]}: {var.x}"
                )
    else:
        print("Optimization ended with status:", model.status)


def main():
    parser = argparse.ArgumentParser(description="Job assignment with learning (Gurobi).")
    parser.add_argument("--data", type=str, default="data/Process_Dataset.xlsx",
                        help="Path to Process_Dataset.xlsx")
    parser.add_argument("--sheet", type=str, default="Part 1",
                        help="Excel sheet name (default: 'Part 1')")
    args = parser.parse_args()

    if not os.path.exists(args.data):
        raise FileNotFoundError(
            f"Excel file not found: {args.data}\n"
            "Put Process_Dataset.xlsx at that path or pass --data <path>."
        )

    solve(args.data, args.sheet)


if __name__ == "__main__":
    main()
