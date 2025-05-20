import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Ensures compatibility on headless servers
import matplotlib.pyplot as plt
import os

def elemK(e):
    L, theta = L_theta(e)
    c, s = np.cos(theta), np.sin(theta)
    k_local = (e[2] * e[3]) / L * np.array([
        [ c*c,  c*s, -c*c, -c*s],
        [ c*s,  s*s, -c*s, -s*s],
        [-c*c, -c*s,  c*c,  c*s],
        [-c*s, -s*s,  c*s,  s*s]
    ])
    return k_local

def L_theta(e):
    n1, n2 = int(e[0])-1, int(e[1])-1
    p1, p2 = Nxy[n1], Nxy[n2]
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    L = np.sqrt(dx**2 + dy**2)
    theta = np.arctan2(dy, dx)
    return L, theta

def globalK(KG, k, e):
    n1, n2 = int(e[0])-1, int(e[1])-1
    dof = [n1*ndof, n1*ndof+1, n2*ndof, n2*ndof+1]
    for i in range(4):
        for j in range(4):
            KG[dof[i], dof[j]] += k[i, j]
    return KG

def truss_solver(node_file, elem_file, ubc_file, fbc_file, result_dir):
    global Nxy, ndof

    nodes_df = pd.read_csv(node_file)
    elem_df = pd.read_csv(elem_file)
    ubc_df = pd.read_csv(ubc_file)
    fbc_df = pd.read_csv(fbc_file)

    Nxy = nodes_df.iloc[:, 1:3].to_numpy(dtype=float)
    numNd = Nxy.shape[0]
    ndof = 2

    elem_data = elem_df.to_numpy()
    if elem_data.shape[1] < 4:
        elem_data = np.hstack((elem_data, np.ones((elem_data.shape[0], 2))))

    max_node_index = int(np.max(elem_data[:, :2]))
    if max_node_index > numNd:
        raise ValueError(f"Element file refers to node {max_node_index}, but only {numNd} nodes are defined.")

    KG = np.zeros((numNd * ndof, numNd * ndof))
    for e in elem_data:
        KG = globalK(KG, elemK(e), e)

    FG = np.zeros(numNd * ndof)
    for i in range(fbc_df.shape[0]):
        node, dirn, val = int(fbc_df.iloc[i, 0])-1, int(fbc_df.iloc[i, 1]), fbc_df.iloc[i, 2]
        FG[node*ndof + dirn] = val

    fixed_dofs = []
    for i in range(ubc_df.shape[0]):
        node, dirn = int(ubc_df.iloc[i, 0])-1, int(ubc_df.iloc[i, 1])
        fixed_dofs.append(node*ndof + dirn)

    free_dofs = list(set(range(numNd * ndof)) - set(fixed_dofs))

    KG_ff = KG[np.ix_(free_dofs, free_dofs)]
    FG_f = FG[free_dofs]

    U = np.zeros(numNd * ndof)
    try:
        U_f = np.linalg.solve(KG_ff, FG_f)
    except np.linalg.LinAlgError as e:
        raise ValueError("Global stiffness matrix is singular. Check constraints or element connectivity.")

    for i, dof in enumerate(free_dofs):
        U[dof] = U_f[i]

    scale = 300  # Scaling factor for displacements
    defNxy = Nxy + scale * U.reshape((numNd, ndof))

    # Plot undeformed truss
    plt.figure()
    for e in elem_data:
        n1, n2 = int(e[0])-1, int(e[1])-1
        x = [Nxy[n1, 0], Nxy[n2, 0]]
        y = [Nxy[n1, 1], Nxy[n2, 1]]
        plt.plot(x, y, 'bo-', linewidth=1)
    plt.title('Undeformed Truss')
    plt.axis('equal')
    plt.grid(True)
    plt.tight_layout()
    undeformed_path = os.path.join(result_dir, 'undeformed.png')
    plt.savefig(undeformed_path)
    plt.close()

    # Plot deformed truss
    plt.figure()
    for e in elem_data:
        n1, n2 = int(e[0])-1, int(e[1])-1
        x = [defNxy[n1, 0], defNxy[n2, 0]]
        y = [defNxy[n1, 1], defNxy[n2, 1]]
        plt.plot(x, y, 'r--', linewidth=2)
    plt.title(f'Deformed Truss (scale={scale}×)')
    plt.axis('equal')
    plt.grid(True)
    plt.tight_layout()
    deformed_path = os.path.join(result_dir, 'deformed.png')
    plt.savefig(deformed_path)
    plt.close()

    # Output displacements to CSV
    output_path = os.path.join(result_dir, 'displacements.csv')
    df_out = pd.DataFrame(U.reshape((numNd, ndof)), columns=['Ux', 'Uy'])
    df_out.to_csv(output_path, index=False)

    return output_path, undeformed_path, deformed_path
