# solver.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def truss_solver(nodes_file, elements_file, ubc_file, fbc_file, output_dir):
    ndof = 2
    Nxy = pd.read_csv(nodes_file, skiprows=1, header=None).values
    elCon = pd.read_csv(elements_file, skiprows=1, header=None).values
    ubc = pd.read_csv(ubc_file, skiprows=1, header=None).values
    fbc = pd.read_csv(fbc_file, skiprows=1, header=None).values

    A = 1
    E = 29.5e6
    numEl = elCon.shape[0]
    numNd = Nxy.shape[0]
    KG = np.zeros((ndof*numNd, ndof*numNd))
    FG = np.zeros(ndof*numNd)

    def elemK(e):
        L, theta = L_theta(e)
        c, s = np.cos(np.radians(theta)), np.sin(np.radians(theta))
        return (E*A/L) * np.array([
            [ c**2,  c*s, -c**2, -c*s],
            [ c*s,  s**2, -c*s, -s**2],
            [-c**2, -c*s,  c**2,  c*s],
            [-c*s, -s**2,  c*s,  s**2]
        ])

    def L_theta(e):
        n1, n2 = int(elCon[e,0])-1, int(elCon[e,1])-1
        p1, p2 = Nxy[n1], Nxy[n2]
        p12 = p2 - p1
        L = np.linalg.norm(p12)
        theta = np.degrees(np.arctan2(p12[1], p12[0]))
        return L, theta

    def globalK(KG, ke, e):
        n1, n2 = int(elCon[e,0])-1, int(elCon[e,1])-1
        DOFs = np.r_[ndof*n1:ndof*(n1+1), ndof*n2:ndof*(n2+1)]
        for i in range(4):
            for j in range(4):
                KG[DOFs[i], DOFs[j]] += ke[i,j]
        return KG

    for e in range(numEl):
        KG = globalK(KG, elemK(e), e)

    KQ = KG.copy()

    for i in range(ubc.shape[0]):
        DOF = int(ndof*(ubc[i,0]-1) + ubc[i,1]-1)
        KG[:, DOF] = 0
        KG[DOF, :] = 0
        KG[DOF, DOF] = 1
        FG[DOF] = ubc[i,2]

    for i in range(fbc.shape[0]):
        DOF = int(ndof*(fbc[i,0]-1) + fbc[i,1]-1)
        FG[DOF] = fbc[i,2]

    U = np.linalg.solve(KG, FG)
    defNxy = Nxy + 300 * U.reshape((numNd, ndof))

    sig, eps = [], []
    for e in range(numEl):
        L, theta = L_theta(e)
        n1, n2 = int(elCon[e,0])-1, int(elCon[e,1])-1
        DOFs = np.r_[ndof*n1:ndof*(n1+1), ndof*n2:ndof*(n2+1)]
        c, s = np.cos(np.radians(theta)), np.sin(np.radians(theta))
        strain = (1/L) * np.array([-c, -s, c, s]) @ U[DOFs]
        eps.append(strain)
        sig.append(E * strain)

    Q = KQ @ U
    result_csv = os.path.join(output_dir, 'truss_results.csv')
    pd.DataFrame({
        'Element': np.arange(1, numEl+1),
        'Stress': sig,
        'Strain': eps,
        'AxialForce': Q[:numEl]
    }).to_csv(result_csv, index=False)

    def plot_truss(xy, title, fname):
        plt.figure()
        for e in range(numEl):
            x = [xy[int(elCon[e,0])-1,0], xy[int(elCon[e,1])-1,0]]
            y = [xy[int(elCon[e,0])-1,1], xy[int(elCon[e,1])-1,1]]
            plt.plot(x, y, 'b-o' if 'undeformed' in fname else 'r-o')
        plt.title(title)
        plt.axis('equal')
        plt.savefig(fname)
        plt.close()

    undeformed_img = os.path.join(output_dir, 'undeformed_truss.png')
    deformed_img = os.path.join(output_dir, 'deformed_truss.png')
    plot_truss(Nxy, 'Undeformed Truss', undeformed_img)
    plot_truss(defNxy, 'Deformed Truss', deformed_img)

    return result_csv, undeformed_img, deformed_img
