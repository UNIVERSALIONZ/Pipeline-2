import pymol
from pymol import cmd
import os
import random

def mutate_residues(protein, chain, residue_positions, mutations):
    cmd.wizard("mutagenesis")
    for residue, mutation in zip(residue_positions, mutations):
        target_residue = f"/{protein}//{chain}/{residue}"
        print(f"Selecting {target_residue} for mutation to {mutation}")
        cmd.get_wizard().do_select(target_residue)
        cmd.get_wizard().set_mode(mutation)
        cmd.frame(1)
        cmd.get_wizard().apply()
    cmd.set_wizard()

def site_saturation_mutagenesis(pdb_file, chain, residue_numbers, output_dir):
    # Load the PDB file
    cmd.load(pdb_file, 'original_protein')
    
    # Define the 20 standard amino acids
    amino_acids = ['ALA', 'CYS', 'ASP', 'GLU', 'PHE', 'GLY', 'HIS', 'ILE', 'LYS', 'LEU',
                   'MET', 'ASN', 'PRO', 'GLN', 'ARG', 'SER', 'THR', 'VAL', 'TRP', 'TYR']
    
    # Generate all combinations of mutations for the two positions
    for aa1 in amino_acids:
        for aa2 in amino_acids:
            # Duplicate the original structure to work on
            cmd.create('protein', 'original_protein')
            
            # Mutate both residues
            mutate_residues('protein', chain, residue_numbers, [aa1, aa2])
            
            # Save the mutated structure
            mutation_desc = f"{chain}{residue_numbers[0]}_{aa1}_{chain}{residue_numbers[1]}_{aa2}"
            output_file = os.path.join(output_dir, f"mutant_{mutation_desc}.pdb")
            cmd.save(output_file, 'protein')
            
            # Delete the modified structure
            cmd.delete('protein')
    
    # Clear the loaded structure
    cmd.delete('all')

# Directory setup
output_dir = 'PDB_Files'
os.makedirs(output_dir, exist_ok=True)

# Example usage
pymol.finish_launching(['pymol', '-cq'])
site_saturation_mutagenesis('pmHAS Mn+2.pdb', 'A', ['130', '132'], output_dir)
cmd.quit()
