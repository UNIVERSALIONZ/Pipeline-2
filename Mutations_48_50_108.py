import os
import csv
from itertools import product
from pymol import cmd, finish_launching

# Define mutation parameters
positions = ['48', '50', '108']
amino_acids = [
    'ALA', 'CYS', 'ASP', 'GLU', 'PHE', 'GLY', 'HIS', 'ILE', 'LYS', 'LEU',
    'MET', 'ASN', 'PRO', 'GLN', 'ARG', 'SER', 'THR', 'VAL', 'TRP', 'TYR'
]
pdb_file = os.path.abspath('pmHAS Mn+2 Edited.pdb')  # Ensure absolute path

# Verify that the PDB file exists
if not os.path.exists(pdb_file):
    raise FileNotFoundError("The PDB file '{}' does not exist.".format(pdb_file))

# Create output directory for mutated PDB files
output_dir = os.path.abspath('PDB_Files')
os.makedirs(output_dir, exist_ok=True)

# Log file for tracking mutation statuses
log_file = os.path.join(output_dir, 'mutation_log.csv')

# Initialize log file if it does not exist
if not os.path.exists(log_file):
    with open(log_file, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Mutation', 'Status'])  # Header

# Function to log mutation status
def log_mutation(mutation_desc, status):
    with open(log_file, 'a') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([mutation_desc, status])

# Generate all combinations of mutations for positions 48, 50, and 108
mutants = [
    '+'.join(["A{}{}".format(pos, aa) for pos, aa in zip(positions, mutation)])
    for mutation in product(amino_acids, repeat=len(positions))
]

# Launch PyMOL
finish_launching(['pymol', '-cq'])

# Process all mutations
for mutation in mutants:
    mutation = mutation.replace(' ', '')
    mutation_list = list(mutation.split('+'))
    residue_list = []
    position_list = []

    for mutation_entry in mutation_list:
        residue = mutation_entry[-3:]  # Last three characters are the amino acid
        position = mutation_entry[1:-3]  # Everything in between is the position
        position_list.append(position)
        residue_list.append(residue)

    try:
        cmd.load(pdb_file, 'original_protein')
        cmd.wizard('mutagenesis')

        for pos, res in zip(position_list, residue_list):
            cmd.get_wizard().do_select('/original_protein//A/{}'.format(pos))
            cmd.get_wizard().set_mode(res)
            cmd.frame(1)
            cmd.get_wizard().apply()

        # Save the mutated structure
        output_file = os.path.join(output_dir, 'mutant_{}.pdb'.format("_".join(mutation_list)))
        cmd.save(output_file, 'original_protein')
        log_mutation(mutation, 'Success')
        print("Mutation {} completed and saved as {}.".format(mutation, output_file))
    except Exception as e:
        log_mutation(mutation, 'Failed: {}'.format(str(e)))
        print("Error processing mutation {}: {}".format(mutation, e))
    finally:
        cmd.reinitialize('everything')

cmd.quit()
