import os
from glob import glob
import subprocess
from subprocess import call

# Paths to MGLTools utilities
mgltools_path = r"C:\Users\NANOPORE\Documents\MGLTools\python.exe"
prepare_receptor_script = r"C:\Users\NANOPORE\Documents\MGLTools\Lib\site-packages\AutoDockTools\Utilities24\prepare_receptor4.py"

# Folder where your minimized PDB files are stored
pdb_input_folder = r"C:\Users\NANOPORE\Documents\Nakul\Data\ML\Pipeline_2\pmHAS_DXD_Mutation_Minimized"
pdbqt_output_folder = r"C:\Users\NANOPORE\Documents\Nakul\Data\ML\Pipeline_2\PDBQT"

# Ensure output folder exists
os.makedirs(pdbqt_output_folder, exist_ok=True)
# Get list of all PDB files in the directory
pdb_files = glob(os.path.join(pdb_input_folder, "*.pdb"))

# Loop over each PDB file and convert to PDBQT format
for pdb_file in pdb_files:
    pdb_filename = os.path.basename(pdb_file)
    pdb_name = os.path.splitext(pdb_filename)[0]

    # Output file in PDBQT format
    pdbqt_file = os.path.join(pdbqt_output_folder, f"{pdb_name}.pdbqt")

    # Command to convert PDB to PDBQT, ensuring non-standard residues like Mn are preserved
    command = f'{mgltools_path} {prepare_receptor_script} -r {pdb_file} -o {pdbqt_file} -A hydrogens -U nphs_lps_waters'

    # Run the command to generate the PDBQT file
    subprocess.run(command, shell=True)

    print(f"Converted {pdb_filename} to PDBQT format and saved as {pdbqt_file}")

print("PDB to PDBQT conversion complete for all structures.")