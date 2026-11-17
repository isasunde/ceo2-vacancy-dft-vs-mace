"""Equation of state of bulk CeO2 at several plane-wave cutoffs.

For each cutoff, energies of the 3-atom primitive cell are computed at a range
of lattice constants and fitted with a Birch-Murnaghan equation of state. The
lattice constant, bulk modulus and energy differences are compared across
cutoffs to decide which cutoff is converged for the properties we report.
"""
import os
import csv

import numpy as np
import matplotlib
matplotlib.use('Agg')   # Headless backend: save figures to file
import matplotlib.pyplot as plt

from gpaw import GPAW, PW
from gpaw.mpi import world
from ase.build import bulk
from ase.eos import EquationOfState
from ase.units import GPa      # eV/Å^3 per GPa, used to convert the bulk modulus

# --- Settings -----------------------
U = 5.0                         # Hubbard U on Ce 4f (eV), same as the scans
XC = 'PBE'
KPTS = 4                        # k-mesh (n x n x n) from the k-point scan

CUTOFFS = [500, 600, 700, 800]  # Cutoffs to compare (eV); the last is the reference
# Lattice constants (Å). The range must bracket the minimum (PBE+U gives
# roughly 5.45-5.50 Å), otherwise the equation-of-state fit is unreliable.
LATTICE_CONSTANTS = np.linspace(5.35, 5.60, 6)

# --- Output directories -------------
os.makedirs('results', exist_ok = True)
os.makedirs('logs', exist_ok = True)

# --- Functions ----------------------
def energy(ecut, a):
    """Return (volume in Å^3, total energy in eV) of the 3-atom CeO2 cell
    (one formula unit) at plane-wave cutoff ecut (eV) and lattice constant a (Å)."""
    atoms = bulk('CeO2', 'fluorite', a = a)

    atoms.calc = GPAW(
        mode = PW(ecut),
        xc = XC,
        setups = {'Ce': f':f,{U}'},         # DFT+U on the Ce f-channel
        kpts = {'size': (KPTS, KPTS, KPTS), 
                'gamma': True
                },
        occupations = {'name': 'fermi-dirac', 
                       'width': 0.05
                       },
        convergence = {'energy': 1e-6},
        txt = f'logs/eos_ecut{ecut}_a{a:.3f}.txt',
    )
    return atoms.get_volume(), atoms.get_potential_energy()

# --- Energy-volume curves and fits --
results = {}    # cutoff -> dict of fitted properties and raw data

for ecut in CUTOFFS:
    vols, es = [], []
    for a in LATTICE_CONSTANTS:
        v, e = energy(ecut, a)
        vols.append(v)
        es.append(e)
        if world.rank == 0:
            print(f'ecut = {ecut} eV, a = {a:.3f} A: {e:.6f} eV', flush = True)

    # Birch-Murnaghan fit: v0 (Å^3), e0 (eV), B (eV/Å^3)
    eos = EquationOfState(vols, es, eos = 'birchmurnaghan')
    v0, e0, B = eos.fit()

    results[ecut] = {
        'a0': (4 * v0) ** (1 / 3),          # fcc primitive cell: V = a^3 / 4
        'B_GPa': B / GPa,
        'e0': e0,
        # Energy cost (eV per formula unit) of compressing / stretching the
        # lattice to the ends of our range, relative to the fitted minimum
        'dE_low': es[0] - e0,
        'dE_high': es[-1] - e0,
        'es': es,
    }

# --- Summary table (relative to the highest cutoff) ----------------------------
if world.rank == 0:
    ref = results[CUTOFFS[-1]]
    header = ['ecut_eV', 'a0_A', 'B_GPa', 'dE_low_meV', 'dE_high_meV',
              'da0_vs_ref_A', 'dB_vs_ref_GPa']
    rows = []
    for ecut in CUTOFFS:
        r = results[ecut]
        rows.append([ecut,
                     round(r['a0'], 4),
                     round(r['B_GPa'], 1),
                     round(r['dE_low'] * 1000, 1),
                     round(r['dE_high'] * 1000, 1),
                     round(r['a0'] - ref['a0'], 4),
                     round(r['B_GPa'] - ref['B_GPa'], 1)])

    print('\n' + '  '.join(f'{h:>13}' for h in header))
    for row in rows:
        print('  '.join(f'{x:>13}' for x in row))

    with open('results/eos_convergence.csv', 'w', newline = '') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

    # --- Plot: energy relative to the fitted minimum vs lattice constant -------
    fig, ax = plt.subplots(figsize = (5.5, 4))
    for ecut in CUTOFFS:
        r = results[ecut]
        ax.plot(LATTICE_CONSTANTS, (np.array(r['es']) - r['e0']) * 1000,
                'o-', label = f'{ecut} eV')
    ax.set_xlabel('Lattice constant (Å)')
    ax.set_ylabel('E - E_min (meV per formula unit)')
    ax.legend(title = 'Cutoff')
    fig.tight_layout()
    fig.savefig('results/eos_convergence.png', dpi = 200)
    plt.close(fig)