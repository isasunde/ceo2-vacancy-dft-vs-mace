import os
import csv

import matplotlib
matplotlib.use("Agg")  # headless backend for the cluster
import matplotlib.pyplot as plt

from gpaw import GPAW, PW
from gpaw.mpi import world
from ase.build import bulk
from ase.io import read, write
from ase.optimize import QuasiNewton

# --- Settings ---------------------
A = 5.41                # Staring lattice constant (Å), based on experimental value
U = 5.0                 # Hubbard U on Ce 4f (eV), fixed during convergence
XC = 'PBE'
CUTOFFS = list(range(400, 851, 50))     # Range of cutoffs to test (eV)
KPOINTS = [2, 3, 4, 5, 6, 7, 8]         # k-points scanned (eV)
ECUT_FIXED = 600                        # cutoff used in k-point scan (eV)
KPTS_FIXED = 4                          # k-mesh used in the cutoff scan (eV)
TOL = 0.001                             # convergenve target (eV/atom = 1 meV/atom)
# ------------------------------------

os.makedirs('results', exist_ok = True)
os.makedirs('logs', exist_ok = True)

# --- Functions ----------------------
def energy_per_atom(ecut, k):
    '''Calculate Potential Energy of CeO2 pr. Atom with ecut & k'''
    atoms = bulk('CeO2', 'fluorite', a = A)     # 3-atom primitive cell
    atoms.calc = GPAW(
        mode = PW(ecut),
        xc = XC,
        setups = {'Ce': f':f,{U}'},
        kpts = {'size': (k, k, k), 
                'gamma': True
                },
        occupations = {'name': 'fermi-dirac', 
                       'width': 0.05
                       },
        convergence = {'energy': 1e-6},
        txt = f'logs/conv_ecut{ecut}:k{k}.txt',
    )
    return atoms.get_potential_energy() / len(atoms)

def save_csv(name, header, rows):
    '''Save result only for one CPU'''
    if world.rank == 0:
        with open(f'results/{name}.csv', 'w', newline = "") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)

def first_converged(xs, es):
    '''First x whose energy stays within the TOL of the final (best)'''
    ref = es[-1]

    for i, x in enumerate(xs):
        if all(abs(e - ref) < TOL for e in es[i:]):
            return x

    return None

def plot(xs, es, xlabel, name):
    if world.rank != 0:
        return

    ref = es[-1]
    de = [(e - ref) * 1000 for e in es]     # Convert to meV/atom

    fig, ax = plt.subplots(figsize = (5, 3.5))
    ax.plot(xs, de, 'o-')
    ax.axhspan(-TOL * 1000, TOL * 1000, alpha = 0.2, label = f'±{TOL*1000:.0f} meV/atom')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('E - E(last) (meV/atom)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(f'results/{name}.png', dpi = 200)
# ------------------------------------

# ---- Cutoff scan -------------------
e_cut = []

for ecut in CUTOFFS:
    e = energy_per_atom(ecut, KPTS_FIXED)
    e_cut.append(e)

    if world.rank == 0:
        print(f'ecut = {ecut} eV, k = {KPTS_FIXED}: {e:.6f} eV/atom', flush=True)

save_csv('cutoff_convergence', ['ecut_eV', 'E_per_atom_eV'], zip(CUTOFFS, e_cut))
plot(CUTOFFS, e_cut, 'Plane-wave cutoff (eV)', 'cutoff_convergence')

# ---- k-points scan -------------------
e_k = []

for k in KPOINTS:
    e = energy_per_atom(ECUT_FIXED, k)
    e_k.append(e)

    if world.rank == 0:
        print(f'ecut = {ECUT_FIXED} eV, k = {k}: {e:.6f} eV/atom', flush=True)

save_csv('kpoint_convergence', ['k', 'E_per_atom_eV'], zip(KPOINTS, e_k))
plot(KPOINTS, e_k, 'k-points per direction', 'kpoint_convergence')

if world.rank == 0:
    print('\nConverged cutoff (eV): ', first_converged(CUTOFFS, e_cut))
    print('\nConverged k-mesh: ', first_converged(KPOINTS, e_k))