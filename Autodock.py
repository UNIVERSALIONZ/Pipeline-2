import os
import subprocess
import pandas as pd
import re

# Define directories
pdbqt_dir = r"C:\Users\NANOPORE\Documents\Nakul\Data\ML\Pipeline_2\DXD_Mutations\Old_Run\PDBQT_Charged"  # Folder containing receptor PDBQT files - Change as per requirement
ligand_file = r"C:\Users\NANOPORE\Documents\Nakul\Data\ML\Pipeline_2\DXD_Mutations\Old_Run\NAG.pdbqt"  # Path to ligand PDBQT file - Change as per requirement
output_dir = r"C:\Users\NANOPORE\Documents\Nakul\Data\ML\Pipeline_2\DXD_Mutations\Old_Run\Docking" # Change as per requirement
log_dir = os.path.join(output_dir, 'Logs')
results_dir = os.path.join(output_dir, 'Outputs')

# Full path to vina executable
vina_path = r"C:\Program Files (x86)\The Scripps Research Institute\Vina\vina.exe"  # Replace with the actual path to vina.exe

# Create output directories if they don't exist
os.makedirs(log_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

# Configuration for AutoDock Vina
center_x, center_y, center_z = 17.779, 14.737, -16.491 # Change as per requirement
size_x, size_y, size_z = 38, 36, 40 # Change as per requirement
energy_range = 4 
exhaustiveness = 8

# Prepare configuration for AutoDock Vina
config_template = f"""
center_x = {center_x}
center_y = {center_y}
center_z = {center_z}
size_x = {size_x}
size_y = {size_y}
size_z = {size_z}
energy_range = {energy_range}
exhaustiveness = {exhaustiveness}
ligand = {ligand_file}
"""

# Run AutoDock Vina docking for each prepared receptor PDBQT
def run_docking(pdbqt_file):
    receptor_pdbqt_path = os.path.join(pdbqt_dir, pdbqt_file)
    output_pdbqt = os.path.join(results_dir, pdbqt_file.replace('.pdbqt', '_out.pdbqt'))
    log_file = os.path.join(log_dir, pdbqt_file.replace('.pdbqt', '_log.txt'))

    # Prepare the config file for docking
    config_file = os.path.join(output_dir, f'config_{pdbqt_file}.txt')
    with open(config_file, 'w', encoding='utf-8') as f:
        f.write(config_template)
        f.write(f"\nreceptor = {receptor_pdbqt_path}\n")

    # Run AutoDock Vina
    vina_cmd = f"\"{vina_path}\" --receptor \"{receptor_pdbqt_path}\" --config \"{config_file}\" --log \"{log_file}\" --out \"{output_pdbqt}\""
    try:
        result = subprocess.run(vina_cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"Standard Output for {pdbqt_file}:\n{result.stdout.decode()}")
        if result.stderr:
            print(f"Standard Error for {pdbqt_file}:\n{result.stderr.decode()}")
    except subprocess.CalledProcessError as e:
        print(f"Error in docking for {pdbqt_file}: {e}")
        if e.stderr:
            print(f"Standard Error: {e.stderr.decode()}")

if __name__ == "__main__":
    pdbqt_files = [f for f in os.listdir(pdbqt_dir) if f.endswith('.pdbqt')]

    # Run docking sequentially for each file
    for pdbqt_file in pdbqt_files:
        run_docking(pdbqt_file)

    # Collect docking results
    results = []
    for pdbqt_file in pdbqt_files:
        log_file = os.path.join(log_dir, pdbqt_file.replace('.pdbqt', '_log.txt'))
        try:
            with open(log_file, 'r') as log:
                for line in log:
                    if "REMARK VINA RESULT:" in line:
                        binding_energy = float(re.findall(r"[-+]?\d*\.\d+|\d+", line)[0])
                        results.append({'Receptor': pdbqt_file, 'Binding Energy (kcal/mol)': binding_energy})
                        break
        except FileNotFoundError:
            print(f"Log file not found for {pdbqt_file}, docking may have failed.")

    # Save results to a CSV file
    if results:
        df = pd.DataFrame(results)
        results_csv = os.path.join(output_dir, 'docking_results.csv')
        df.to_csv(results_csv, index=False)
        print(f"Results saved to {results_csv}")
    else:
        print("No docking results to save.")
