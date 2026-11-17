"""Cutoff and k-point convergence tests for bulk CeO2 with GPAW.

Optimum scans on the 3-atom primitive cell:
  1. Plane-wave cutoff, at a fixed k-mesh.
  2. k-mesh, at a fixed plane-wave cutoff.

"""
import os
import csv

import matplotlib
matplotlib.use('Agg')   # Headless backend: save figures to file, never open a window
import matplotlib.pyplot as plt

from gpaw import GPAW, PW
from gpaw.mpi import world
from ase.build import bulk

# --- Settings ---------------------
A = 5.41                # Starting lattice constant (Å), based on experimental value
U = 5.0                 # Hubbard U on Ce 4f (eV), fixed during convergence
XC = 'PBE'              # Exchange-correlation functional

CUTOFFS = list(range(400, 1001, 50))     # Range of cutoffs to test (eV)
KPOINTS = [2, 3, 4, 5, 6, 7, 8, 9, 10]         # Range of k-points to test (eV)

ECUT_FIXED = 700                        # Cutoff fixed in k-point scan (eV)
KPTS_FIXED = 4                          # k-mesh fixed in the cutoff scan (eV)

TOL = 0.001                             # Convergence target (eV/atom = 1 meV/atom)

# --- Output directories -------------
os.makedirs('results', exist_ok = True)
os.makedirs('logs', exist_ok = True)

# --- Functions ----------------------
def energy_per_atom(ecut, k):
    """Return the DFT total energy per atom (eV/atom) of CeO2.

    ecut : plane-wave cutoff (eV)
    k    : k-points per direction, giving a k x k x k Γ-centred mesh
    """
    # Fluorite CeO2 as the 3-atom primitive cell
    atoms = bulk('CeO2', 'fluorite', a = A)  

    atoms.calc = GPAW(
        mode = PW(ecut),
        xc = XC,
        # DFT+U: ':f,U' applies the Hubbard correction to the Ce f-channel,
        # which corrects the self-interaction error on the localised 4f states
        setups = {'Ce': f':f,{U}'},
        # Γ-centred mesh (includes the k = 0 point)
        kpts = {'size': (k, k, k), 
                'gamma': True
                },
        # Small electronic smearing helps SCF convergence; at 0.05 eV it has a
        # negligible effect on the energy of an insulator like CeO2
        occupations = {'name': 'fermi-dirac', 
                       'width': 0.05
                       },
        # SCF stops when the energy changes by less than 1e-6 eV
        # (tighter than TOL, so SCF noise does not hide the trend)
        convergence = {'energy': 1e-6},
        txt = f'logs/conv_ecut{ecut}_k{k}.txt',     # one log per calculation
    )
    # Divide by atom count so results are comparable between cell sizes
    return atoms.get_potential_energy() / len(atoms)

def save_csv(name, header, rows):
    """Write results to results/<name>.csv 
    (rank 0 only, to enable MPI parallization)."""
    if world.rank == 0:
        with open(f'results/{name}.csv', 'w', newline = "") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)

def first_converged(xs, es):
    """Return the first setting x for which every later step changes the
    energy by less than TOL; None if no setting qualifies (extend range).

    The changes at x is |E(x + 1) - E(x)|, i.e. what you would still gain by
    increasing the setting by one step

    NOTE: The last point has no 'next' step and can therefore not be converged.
    """
    steps = [abs(es[i + 1] - es[i]) for i in range(len(es) - 1)]

    for i, x in enumerate(xs[:-1]):
        if all(s < TOL for s in steps[i:]):
            return x

    return None

def plot(xs, es, xlabel, name):
    """Plot the energy change relative to the next point (meV/atom, log scale) against
    the setting, with the TOL line dashed, and save to results/ (rank 0 only).
    
    A setting is converged once its point, and all points to its right,
    lie below the dashed line."""
    if world.rank != 0:
        return

    steps = [abs(es[i + 1] - es[i]) * 1000 for i in range(len(es) - 1)]   # meV/atom

    fig, ax = plt.subplots(figsize = (5, 3.5))
    ax.semilogy(xs[:-1], steps, 'o-')
    ax.axhline(TOL * 1000, ls = '--', color = 'gray', label = f'TOL = {TOL*1000:.0f} meV/atom')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('|E(x + 1) - E(x)| (meV/atom)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(f'results/{name}.png', dpi = 200)
    plt.close(fig)
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