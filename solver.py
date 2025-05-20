import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def elemK(e, A=1, E=200e9):
    L, theta = L_theta(e)
    c, s = np.cos(theta), np.sin(theta)
    k = A * E / L
    k_local = k * np.array([
        [ c*c,  c*s, -c*c, -c*s],
        [ c*s,  s*s, -c*s, -s*s],
        [-c*c, -c*s,  c*c,  c*s],
        [-c*s, -s*s,  c*s,  s*s]
    ])
    return k_local

def L_theta(e):
    n1, n2 = int(e[0])-1, int(e[1])-1
    p1, p2 = Nxy[n1], Nxy[n2]
    L = np.linalg.norm(p2 - p1)
    theta = np.arctan2(p2[1] - p1[1], p2[0] - p1[0])
    return L, theta

def globalK(KG, k, e):
    nd = ndof * (int(e[0])-1)
    ni = [nd, nd+1, ndof*(int(e[1])-1), ndof*(int(e[1])-1)+1]
    for i in range(4):
        for j in range(4):
            KG[ni[i], ni[j]] += k[i, j]
    return KG

def truss_solver(node_file, element_file, ubc_file, fbc_file, result_dir):
    global ndof, Nxy
    ndof = 2

    # Read input files
    Ndf = pd.read_csv(node_file).dropna().values
    Edf = pd.read_csv(element_file).dropna().values
    Udf = pd.read_csv(ubc_file).dropna().values
    Fdf = pd.read_csv(fbc_file).dropna().values

    # Node coordinates
    Nxy = Ndf[:, 1:3]
    numNd = Nxy.shape[0]

    # Element connectivity
    elCon = Edf[:, 1:3].astype(int)
    max_node_index = int(np.max(elCon))
    if max_node_index > numNd:
        raise ValueError(f"Element file refers to node {max_node_index}, but only {numNd} nodes are defined.")

    # Global stiffness matrix
    KG = np.zeros((ndof*numNd, ndof*numNd))
    for e in elCon:
        KG = globalK(KG, elemK(e), e)

    # Force vector
    FG = np.zeros((ndof*numNd,))
    for f in Fdf:
        node = int(f[0]) - 1
        dof = int(f[1])
        val = f[2]
        FG[node*ndof + dof] = val

    # Boundary conditions
    bc = []
    for u in Udf:
        node = int(u[0]) - 1
        dof = int(u[1])
        bc.append(node*ndof + dof)
    bc = np.array(bc)

    # Solve for displacements
    free_dofs = np.setdiff1d(np.arange(ndof*numNd), bc)
    KG_reduced = KG[np.ix_(free_dofs, free_dofs)]
    FG_reduced = FG[free_dofs]
    U = np.zeros((ndof*numNd,))
    U[free_dofs] = np.linalg.solve(KG_reduced, FG_reduced)

    # Save results to CSV
    disp = U.reshape((numNd, ndof))
    output_csv = os.path.join(result_dir, 'displacements.csv')
    pd.DataFrame(disp, columns=['Ux', 'Uy']).to_csv(output_csv, index=False)

    # Plot undeformed truss
    undeformed_img = os.path.join(result_dir, 'undeformed.png')
    plt.figure()
    for e in elCon:
        i, j = int(e[0])-1, int(e[1])-1
        x = [Nxy[i, 0], Nxy[j, 0]]
        y = [Nxy[i, 1], Nxy[j, 1]]
        plt.plot(x, y, 'k-', lw=1)
    plt.title("Undeformed Truss")
    plt.axis('equal')
    plt.savefig(undeformed_img)
    plt.close()

    # Plot deformed truss (scaled)
    defNxy = Nxy + 300 * disp  # 300 is a scale factor for visualization
    deformed_img = os.path.join(result_dir, 'deformed.png')
    plt.figure()
    for e in elCon:
        i, j = int(e[0])-1, int(e[1])-1
        x = [defNxy[i, 0], defNxy[j, 0]]
        y = [defNxy[i, 1], defNxy[j, 1]]
        plt.plot(x, y, 'r--', lw=1)
    plt.title("Deformed Truss")
    plt.axis('equal')
    plt.savefig(deformed_img)
    plt.close()

    return output_csv, undeformed_img, deformed_img
