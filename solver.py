import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def truss_solver(nodes_path, elements_path, ubc_path, fbc_path, result_dir):
    # Load input files
    nodes_df = pd.read_csv(nodes_path)
    elements_df = pd.read_csv(elements_path)
    ubc_df = pd.read_csv(ubc_path)
    fbc_df = pd.read_csv(fbc_path)

    Nxy = nodes_df[['X', 'Y']].values
    numNd = len(Nxy)
    ndof = 2
    dof = numNd * ndof

    # Check for mismatches
    max_node_index = max(elements_df[['Node1', 'Node2']].values.flatten())
    if max_node_index >= numNd:
        raise ValueError(f"Element file refers to node {max_node_index}, but only {numNd} nodes are defined.")

    # Initialize global stiffness matrix and force vector
    KG = np.zeros((dof, dof))
    FG = np.zeros((dof, 1))

    # Assemble global stiffness matrix
    def L_theta(e):
        n1, n2 = int(e[0]), int(e[1])
        p1, p2 = Nxy[n1], Nxy[n2]
        dx, dy = p2 - p1
        L = np.hypot(dx, dy)
        theta = np.arctan2(dy, dx)
        return L, theta

    def elemK(e):
        A, E = e[2], e[3]
        L, theta = L_theta(e)
        c, s = np.cos(theta), np.sin(theta)
        k = (A * E / L) * np.array([
            [ c*c,  c*s, -c*c, -c*s],
            [ c*s,  s*s, -c*s, -s*s],
            [-c*c, -c*s,  c*c,  c*s],
            [-c*s, -s*s,  c*s,  s*s]
        ])
        return k

    def globalK(KG, k, e):
        n1, n2 = int(e[0]), int(e[1])
        dof_map = [n1*2, n1*2+1, n2*2, n2*2+1]
        for i in range(4):
            for j in range(4):
                KG[dof_map[i], dof_map[j]] += k[i, j]
        return KG

    for _, e in elements_df.iterrows():
        KG = globalK(KG, elemK(e), e)

    # Apply forces
    for _, f in fbc_df.iterrows():
        node = int(f['Node'])
        dof_x, dof_y = node * 2, node * 2 + 1
        FG[dof_x] += f['Fx']
        FG[dof_y] += f['Fy']

    # Apply boundary conditions
    fixed_dofs = []
    for _, bc in ubc_df.iterrows():
        node = int(bc['Node'])
        if bc['UX'] == 0:
            fixed_dofs.append(node * 2)
        if bc['UY'] == 0:
            fixed_dofs.append(node * 2 + 1)

    free_dofs = np.setdiff1d(np.arange(dof), fixed_dofs)

    K_ff = KG[np.ix_(free_dofs, free_dofs)]
    F_f = FG[free_dofs]

    # Solve displacements
    try:
        U_f = np.linalg.solve(K_ff, F_f)
    except np.linalg.LinAlgError as e:
        raise ValueError("Stiffness matrix is singular. Check support and element configuration.")

    U = np.zeros((dof, 1))
    U[free_dofs] = U_f.reshape(-1, 1)

    # Compute deformed shape
    defNxy = Nxy + 300 * U.reshape((numNd, ndof))

    # Plot undeformed truss
    plt.figure(figsize=(10, 5))
    for _, e in elements_df.iterrows():
        n1, n2 = int(e['Node1']), int(e['Node2'])
        plt.plot([Nxy[n1][0], Nxy[n2][0]], [Nxy[n1][1], Nxy[n2][1]], 'bo-')
    undeformed_img = os.path.join(result_dir, "undeformed.png")
    plt.title("Undeformed Truss")
    plt.savefig(undeformed_img)
    plt.close()

    # Plot deformed truss
    plt.figure(figsize=(10, 5))
    for _, e in elements_df.iterrows():
        n1, n2 = int(e['Node1']), int(e['Node2'])
        plt.plot([defNxy[n1][0], defNxy[n2][0]], [defNxy[n1][1], defNxy[n2][1]], 'ro-')
    deformed_img = os.path.join(result_dir, "deformed.png")
    plt.title("Deformed Truss")
    plt.savefig(deformed_img)
    plt.close()

    # Prepare output CSV with displacements
    disp = U.reshape((numNd, ndof))
    nodes_df['UX'] = disp[:, 0]
    nodes_df['UY'] = disp[:, 1]
    output_csv = os.path.join(result_dir, "output.csv")
    nodes_df.to_csv(output_csv, index=False)

    return output_csv, undeformed_img, deformed_img
