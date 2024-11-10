import os
from pyrosetta import *
from rosetta import *
import glob
import matplotlib.pyplot as plt

init()  # Initialize Rosetta only once

input_dir = "pmHAS_DXD_400"
output_dir = "pmHAS_DXD_Mutation_Minimized"
os.makedirs(output_dir, exist_ok=True)

files = glob.glob(os.path.join(input_dir, "*"))

scorefxn = get_fa_scorefxn()  # Initialize the scoring function once

initial_energies=[]
final_energies=[]
strc_name = []

for file in files:
    p = Pose()
    p = pose_from_pdb(file)

    initial_energy = scorefxn(p)
    initial_energies.append(initial_energy)
    strc_name.append(os.path.basename(file))


    ncycles = 50
    kT = 1.0
    mc = MonteCarlo(p, scorefxn, kT)

    movemap = MoveMap()
    movemap.set_bb(True)

    small_mover = pyrosetta.rosetta.protocols.simple_moves.SmallMover(movemap, kT, 5)

    for _ in range(ncycles):
        small_mover.apply(p)
        mc.boltzmann(p)
    
    final_energy = scorefxn(p)
    final_energies.append(final_energy)

    mc.recover_low(p)
    output_file = os.path.join(output_dir, os.path.basename(file))
    dump_pdb(p, output_file)



plt.figure(figsize = (10,6))
plt.plot(structure_names, inital_energies, label = "Initial energy", marker = 'o')
plt.plot(structure_names, final_energies, label = "Final energy", marker = 'x')
plt.xlabel("Structure")
plt.ylabel("Energy")
plt.title("Energy Before and After Minimization")
plt.xticks(rotation=90)
plt.legend()
plt.tight_layout()
plt.savefig("Energy.png")
plt.show()