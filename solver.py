# solver.py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

def truss_solver(nodes_file, elements_file, ubc_file, fbc_file, output_dir):
    # Read inputs
    nodes = pd.read_csv(nodes_file)
    elements = pd.read_csv(elements_file)
    ubc = pd.read_csv(ubc_file)
    fbc = pd.read_csv(fbc_file)

    # Mock output for demonstration
    displacements = pd.DataFrame({
        'Node': nodes['Node'],
        'X_Displacement': np.random.randn(len(nodes)) * 0.01,
        'Y_Displacement': np.random.randn(len(nodes)) * 0.01
    })

    output_csv_path = os.path.join(output_dir, 'displacements.csv')
    displacements.to_csv(output_csv_path, index=False)

    def plot_truss(nodes, elements, displacements=None, title='', filename='plot.png'):
        plt.figure(figsize=(8, 6))
        for _, row in elements.iterrows():
            n1 = nodes[nodes['Node'] == row['Node1']].iloc[0]
            n2 = nodes[nodes['Node'] == row['Node2']].iloc[0]
            x = [n1['X'], n2['X']]
            y = [n1['Y'], n2['Y']]
            plt.plot(x, y, 'wo-')
        if displacements is not None:
            scale = 100
            for _, row in elements.iterrows():
                n1 = nodes[nodes['Node'] == row['Node1']].iloc[0]
                n2 = nodes[nodes['Node'] == row['Node2']].iloc[0]
                d1 = displacements[displacements['Node'] == row['Node1']].iloc[0]
                d2 = displacements[displacements['Node'] == row['Node2']].iloc[0]
                x = [n1['X'] + d1['X_Displacement'] * scale, n2['X'] + d2['X_Displacement'] * scale]
                y = [n1['Y'] + d1['Y_Displacement'] * scale, n2['Y'] + d2['Y_Displacement'] * scale]
                plt.plot(x, y, 'ro-')
        plt.title(title)
        plt.gca().set_facecolor('black')
        plt.grid(True, color='gray')
        plt.savefig(os.path.join(output_dir, filename), facecolor='black')
        plt.close()

    plot_truss(nodes, elements, None, title='Undeformed Truss', filename='undeformed.png')
    plot_truss(nodes, elements, displacements, title='Deformed Truss', filename='deformed.png')
    plot_truss(nodes, elements, displacements, title='Interactive View', filename='interactive.png')

    return (
        output_csv_path,
        os.path.join(output_dir, 'undeformed.png'),
        os.path.join(output_dir, 'deformed.png'),
        os.path.join(output_dir, 'interactive.png')
    )
