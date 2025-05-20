import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO

def L_theta(e):
    n1, n2, A, E = e
    p1, p2 = Nxy[n1], Nxy[n2]
    dx, dy = p2 - p1
    L = np.sqrt(dx**2 + dy**2)
    theta = np.arctan2(dy, dx)
    return L, theta

def elemK(e):
    L, theta = L_theta(e)
    c, s = np.cos(theta), np.sin(theta)
    A, E = e[2], e[3]
    k = A * E / L
    return k * np.array([
        [ c*c,  c*s, -c*c, -c*s],
        [ c*s,  s*s, -c*s, -s*s],
        [-c*c, -c*s,  c*c,  c*s],
        [-c*s, -s*s,  c*s,  s*s]
    ])

def globalK(KG, k, e):
    n1, n2 = int(e[0]), int(e[1])
    dof = [n1*2, n1*2+1, n2*2, n2*2+1]
    for i in range(4):
        for j in range(4):
            KG[dof[i], dof[j]] += k[i, j]
    return KG

def plot_truss(nodes, elements, title):
    fig, ax = plt.subplots()
    for e in elements:
        n1, n2 = int(e[0]), int(e[1])
        x = [nodes[n1][0], nodes[n2][0]]
        y = [nodes[n1][1], nodes[n2][1]]
        ax.plot(x, y, 'b-o', linewidth=1)
    ax.set_title(title)
    ax.axis('equal')
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close()
    return buffer

def truss_solver(nodes_df, elements_df, fbc_df, ubc_df):
    global Nxy

    Nxy = nodes_df[['x', 'y']].values
    numNd, ndof = Nxy.shape
    dof = ndof * numNd

    elements = elements_df[['node1', 'node2', 'A', 'E']].values
    max_node_index = int(np.max(elements[:, :2]))
    if max_node_index >= numNd:
        raise ValueError(f"Element file refers to node {max_node_index}, but only {numNd} nodes are defined.")

    KG = np.zeros((dof, dof))
    for e in elements:
        KG = globalK(KG, elemK(e), e)

    FG = np.zeros((dof,))
    for _, row in fbc_df.iterrows():
        node, fx, fy = int(row['node']), row['fx'], row['fy']
        FG[node*2] = fx
        FG[node*2+1] = fy

    known_dof = []
    for _, row in ubc_df.iterrows():
        node, ux, uy = int(row['node']), row['ux'], row['uy']
        if ux == 0:
            known_dof.append(node*2)
        if uy == 0:
            known_dof.append(node*2+1)

    all_dof = np.arange(dof)
    free_dof = np.setdiff1d(all_dof, known_dof)

    KFF = KG[np.ix_(free_dof, free_dof)]
    FFF = FG[free_dof]

    try:
        UFF = np.linalg.solve(KFF, FFF)
    except np.linalg.LinAlgError:
        raise ValueError("Stiffness matrix is singular. Check constraints or element connectivity.")

    U = np.zeros((dof,))
    U[free_dof] = UFF

    # ================================
    # Deformation scaling for plotting
    # ================================
    U_reshaped = U.reshape((numNd, ndof))
    max_disp = np.max(np.abs(U_reshaped))
    model_size = np.max(np.linalg.norm(Nxy, axis=1))
    scale_factor = 0.1 * model_size / max_disp if max_disp > 0 else 1
    defNxy = Nxy + scale_factor * U_reshaped

    # Output displacements CSV
    output = pd.DataFrame({
        'node': range(numNd),
        'ux': U[0::2],
        'uy': U[1::2]
    })

    # Plot images
    undeformed_img = plot_truss(Nxy, elements, "Undeformed Truss")
    deformed_img = plot_truss(defNxy, elements, "Deformed Truss")

    return output, undeformed_img, deformed_img
