"""Build the 2x2x2 CeO2 supercell with one oxygen vacancy and check it."""
import os
import io

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from PIL import Image
from ase import Atoms
from ase.io import write
from ase.build import bulk
from ase.geometry import get_distances
from ase.data.colors import jmol_colors
from ase.visualize.plot import plot_atoms

# --- Settings -----------------------
A0 = 5.470      # Lattice constant (Å): PBE+U (U = 5 eV) EOS fit at 700 ev

# --- Output directories -------------
os.makedirs('results', exist_ok = True)

# --- Build ---------------------------
# Setup supercell of 24 atoms
perfect = bulk('CeO2', 'fluorite', a = A0).repeat((2, 2, 2))

o_idx = [a.index for a in perfect if a.symbol == 'O']
ce_idx = [a.index for a in perfect if a.symbol == 'Ce']

# Select O to set as vacant (All O sites are equal in fluorite)
vac = o_idx[0]
vac_pos = perfect.positions[vac].copy()

# Remove O at vacancy
defect = perfect.copy()
del defect[vac]

# --- Numerical checks ----------------
print(f'Perfect: {perfect.get_chemical_formula()} ({len(perfect)} atoms)')
print(f'Defect: {defect.get_chemical_formula()} ({len(defect)} atoms)')
print(f'Vacancy concentration: 1 of {len(o_idx)} O sites = {100 / len(o_idx):.2f} %')
print(f'Supercell lengths (Å): {np.round(defect.cell.lengths(), 3)}'
      f'  -> vacancy-image distance = {defect.cell.lengths().min():.2f} Å')

# Calculate distances from the empty site to every atom
_, dist = get_distances(np.atleast_2d(vac_pos), 
                       defect.positions,
                       cell = defect.cell,
                       pbc = True
                       )
order = np.argsort(dist[0])
print('\nNeighbours of the vacancy site (symbol, distance):')
for i in order[:10]:
    print(f'    {defect[i].symbol:2s} {dist[0][i]:.3f} Å')
print(f'Expected: 4 Ce at {A0 * np.sqrt(3)/4:.3f} Å, 6 O at {A0/2:.3f} Å')

# --- Visualize the cell with vacancy highlighted ---
# Create defect with placeholder fro vacancy
vis = defect.copy()
vis += Atoms('X', positions = [vac_pos])

# Setup colors (pale gray for vacancy)
colors = [jmol_colors[n] for n in vis.numbers]
colors[-1] = (0.85, 0.85, 0.85)

radii = [0.7 if s == 'Ce' else 0.4 for s in vis.get_chemical_symbols()]

fig, axes = plt.subplots(1, 2, figsize = (8, 4))
for ax, rot in zip(axes, ['-70x, -20y, 0z', '0x, 0y, 0z']):
    plot_atoms(vis, ax, 
               rotation = rot, 
               radii = radii,
               colors = colors,
               show_unit_cell = 2
               )
    ax.set_axis_off()

fig.tight_layout()
fig.savefig('results/CeO2_vacancy_structure.png', dpi = 200)
plt.close(fig)

# ---- Rotating GIF -------------------
frames = []

for ang in range(0, 360, 5):
    fig, ax = plt.subplots(figsize=(3, 3))
    plot_atoms(vis, ax, 
               rotation = f'0x, 0y, {ang}z', 
               radii = radii,
               colors = colors,
               show_unit_cell = 2
               )
    ax.set_axis_off()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=80)
    plt.close(fig)
    buf.seek(0)
    frames.append(Image.open(buf).convert("RGB"))
frames[0].save("results/CeO2_vacancy_rotation.gif", save_all=True,
               append_images=frames[1:], duration=400, loop=0)

write('results/CeO2_vacancy.xyz', vis)      # Includes the X marker; open in VESTA/OVITO
write('results/CeO2_perfect_2x2x2.cif', perfect)
