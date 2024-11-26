import os
import glob
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from pyrosetta import *
from rosetta import *

init()  # Initialize Rosetta only once

input_dir = "pmHAS_DXD_400"
output_dir = "pmHAS_DXD_Mutation_Minimized_3_wl"
os.makedirs(output_dir, exist_ok=True)

files = glob.glob(os.path.join(input_dir, "*"))

# Score function
scorefxn = get_fa_scorefxn()

# Function to process each file
def process_file(file):
    try:
        # Load the PDB file into a Pose object
        p = pose_from_pdb(file)
    except Exception as e:
        print(f"Error loading {file}: {e}")
        return None  # Return None if an error occurs

    # Calculate initial energy
    initial_energy = scorefxn(p)
    structure_name = os.path.basename(file)

    # Monte Carlo energy minimization
    ncycles = 10000
    kT = 1.0
    mc = MonteCarlo(p, scorefxn, kT)

    movemap = MoveMap()
    movemap.set_bb(True)

    small_mover = pyrosetta.rosetta.protocols.simple_moves.SmallMover(movemap, kT, 5)

    for _ in range(ncycles):
        small_mover.apply(p)
        mc.boltzmann(p)

    mc.recover_low(p)
    
    # Calculate final energy
    final_energy = scorefxn(p)

    # Save the minimized structure
    output_file = os.path.join(output_dir, structure_name)
    p.dump_pdb(output_file)

    return (structure_name, initial_energy, final_energy)

# Set the number of cores
num_cores = 10

# Use multiprocessing to process files in parallel
with Pool(num_cores) as pool:
    results = pool.map(process_file, files)

# Filter out any None results (from errors)
results = [result for result in results if result is not None]

# Unpack results
structure_names, initial_energies, final_energies = zip(*results)

# Plot the energies
plt.figure(figsize=(10, 6))
plt.plot(structure_names, initial_energies, label="Initial Energy", marker='o')
plt.plot(structure_names, final_energies, label="Final Energy", marker='x')
plt.xlabel("Structure")
plt.ylabel("Energy")
plt.title("Energy Before and After Minimization")
plt.xticks(rotation=90)
plt.legend()
plt.tight_layout()
plt.savefig("energy_comparison.png")
plt.show()
