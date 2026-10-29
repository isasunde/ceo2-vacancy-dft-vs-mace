"""Build CeO2 (fluorite) and check it visually and numerically. """
import io
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from ase.build import bulk
from ase.io import write
from ase.neighborlist import neighbor_list
from ase.visualize.plot import plot_atoms

A = 5.41
os.makedirs("results", exist_ok=True)

prim = bulk("CeO2", "fluorite", a=A)               # 3 atoms, used for DFT
cubic = bulk("CeO2", "fluorite", a=A, cubic=True)  # 12 atoms, easier to look at

# ---- 1. Numerical sanity checks -------------------------------------
for name, at in [("primitive", prim), ("cubic", cubic)]:
    print(f"{name}: {len(at)} atoms, {at.get_chemical_formula()}, "
          f"volume = {at.get_volume():.2f} A^3")

i, j, d = neighbor_list("ijd", cubic, 2.6)   # cutoff catches only Ce-O bonds
symbols = cubic.get_chemical_symbols()

for idx in range(len(cubic)):
    dists = d[i == idx]
    print(f"{symbols[idx]:2s} atom {idx:2d}: {len(dists)} neighbours, "
          f"distance(s) {np.unique(np.round(dists, 3))} A")

print(f"Expected Ce-O distance: {A * np.sqrt(3) / 4:.3f} A")

# ---- 2. Static views from three directions ----------------------------
def radii_for(at):
    return np.array([0.7 if s == "Ce" else 0.4 for s in at.get_chemical_symbols()])

views = {"top (along c)": "0x,0y,0z",
         "side (along b)": "-90x",
         "tilted": "-70x,-20y"}

fig, axes = plt.subplots(1, 3, figsize=(10, 4.5))

for ax, (title, rot) in zip(axes, views.items()):
    plot_atoms(cubic, ax, rotation=rot, radii=radii_for(cubic), show_unit_cell=2)
    ax.set_title(title)
    ax.set_axis_off()

fig.tight_layout()
fig.savefig("results/CeO2_structure.png", dpi=200)
plt.close(fig)

# ---- 3. Rotating GIF ---------------------------------------------------
frames = []

for ang in range(0, 360, 15):
    fig, ax = plt.subplots(figsize=(3, 3))
    plot_atoms(cubic, ax, rotation=f"{ang}z,-70x", radii=radii_for(cubic),
               show_unit_cell=2)
    ax.set_axis_off()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=80)
    plt.close(fig)
    buf.seek(0)
    frames.append(Image.open(buf).convert("RGB"))
frames[0].save("results/CeO2_rotation.gif", save_all=True,
               append_images=frames[1:], duration=300, loop=0)

# ---- 4. Files for a proper 3D viewer ----------------------------------
write("results/CeO2_cubic.cif", cubic)
write("results/CeO2_primitive.cif", prim)
print("Wrote results/CeO2_structure.png, CeO2_rotation.gif and .cif files")