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

## To remove the series of text towards the end of the file, use this code.
# Path to the folder containing the original PDB files
pdb_folder = '/media/sudipta/638a7113-11e4-49cc-8a8f-e369d3ce1bd1/ML_PI/Pipeline_2/10000STEPS/10000PDBs_striped'

# Path to the folder where the modified PDB files will be saved
output_folder = '/media/sudipta/638a7113-11e4-49cc-8a8f-e369d3ce1bd1/ML_PI/Pipeline_2/10000STEPS/10000PDBs_striped_MN'

# Make sure the output folder exists, if not, create it
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Function to process a single PDB file and replace MG with MN
def replace_mg_with_mn(input_file, output_file):
    with open(input_file, 'r') as file:
        # Read the content of the file
        content = file.read()
    
    # Replace 'MG' with 'MN'
    modified_content = content.replace('MG', 'MN')
    
    # Write the modified content to the output file
    with open(output_file, 'w') as file:
        file.write(modified_content)

# Loop through all PDB files in the folder
for pdb_file in os.listdir(pdb_folder):
    if pdb_file.endswith('.pdb'):  # Only process PDB files
        pdb_path = os.path.join(pdb_folder, pdb_file)
        
        # Define the output file path
        output_pdb_path = os.path.join(output_folder, pdb_file)
        
        # Process the PDB file and replace MG with MN
        replace_mg_with_mn(pdb_path, output_pdb_path)

print(f"Processing complete. Modified PDB files saved in: {output_folder}")
