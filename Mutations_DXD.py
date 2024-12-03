import os
import subprocess
import csv
from itertools import product

# Define the directory to save PDB files
output_dir = 'PDB_Files'
os.makedirs(output_dir, exist_ok=True)

# Define mutation parameters
chain = 'A'
residues = ['130', '132']
amino_acids = [
    'ALA', 'CYS', 'ASP', 'GLU', 'PHE', 'GLY', 'HIS', 'ILE', 'LYS', 'LEU',
    'MET', 'ASN', 'PRO', 'GLN', 'ARG', 'SER', 'THR', 'VAL', 'TRP', 'TYR'
]

# Log file for tracking mutation statuses
log_file = 'mutation_log.csv'

# Function to run a mutation in a separate PyMOL instance
def run_mutation(pdb_file, chain, residue1, residue2, aa1, aa2, output_file):
    pymol_script = f"""
import pymol
from pymol import cmd
cmd.load('{pdb_file}', 'original_protein')
cmd.wizard('mutagenesis')
cmd.get_wizard().do_select('/original_protein//{chain}/{residue1}')
cmd.get_wizard().set_mode('{aa1}')
cmd.frame(1)
cmd.get_wizard().apply()
cmd.get_wizard().do_select('/original_protein//{chain}/{residue2}')
cmd.get_wizard().set_mode('{aa2}')
cmd.frame(1)
cmd.get_wizard().apply()
cmd.set_wizard()
cmd.save('{output_file}', 'original_protein')
cmd.quit()
"""
    script_path = "temp_mutation_script.pml"
    with open(script_path, "w") as file:
        file.write(pymol_script)
    subprocess.run(["pymol", "-cq", script_path])
    os.remove(script_path)

# Initialize log file if it does not exist
if not os.path.exists(log_file):
    with open(log_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Mutation', 'Residue1', 'Residue2', 'Status'])  # Header

# Function to check if a mutation was previously processed
def is_logged(mutation_desc):
    with open(log_file, 'r') as csvfile:
        reader = csv.reader(csvfile)
        return any(row[0] == mutation_desc and row[3] == 'Success' for row in reader)

# Batch processing to avoid memory overload and log progress
batch_size = 50  # Adjust batch size as needed
mutation_combinations = list(product(amino_acids, repeat=2))

# Process each mutation combination for 400 mutations
for i, (aa1, aa2) in enumerate(mutation_combinations[:400]):
    mutation_desc = f"{chain}{residues[0]}_{aa1}_{chain}{residues[1]}_{aa2}"
    output_file = os.path.join(output_dir, f"mutant_{mutation_desc}.pdb")

    # Check if mutation is already logged and skip if successful
    if is_logged(mutation_desc):
        print(f"{mutation_desc} already processed, skipping.")
        continue

    print(f"Processing mutation {mutation_desc}")

    try:
        # Run the mutation in a separate PyMOL instance
        run_mutation('pmHAS Mn+2 Edited.pdb', chain, residues[0], residues[1], aa1, aa2, output_file)
        
        # Verify if the file was created successfully
        if os.path.exists(output_file):
            # Log the successful mutation only after verification
            with open(log_file, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([mutation_desc, aa1, aa2, 'Success'])
            print(f"Mutation {mutation_desc} completed and verified.")
        else:
            # Log as failed if the file is missing
            print(f"Error: File for mutation {mutation_desc} was not created.")
            with open(log_file, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([mutation_desc, aa1, aa2, 'Failed'])

    except Exception as e:
        print(f"Error processing {mutation_desc}: {e}")
        
        # Log the failed mutation attempt
        with open(log_file, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([mutation_desc, aa1, aa2, 'Failed'])
    
    # Batch reset to free memory
    if (i + 1) % batch_size == 0:
        print(f"Batch {i // batch_size + 1} processed. Restarting PyMOL...")
        subprocess.run(["pymol", "-cq"])  # Reset PyMOL
