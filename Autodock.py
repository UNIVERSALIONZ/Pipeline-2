import os
import subprocess
import pandas as pd
import re

# Define directories
pdbqt_dir = 'path/to/your/mutants_pdbqt_folder'  # Folder containing receptor PDBQT files
ligand_file = 'path/to/n_cetylglucosamine.pdbqt'  # Path to ligand PDBQT file
output_dir = 'path/to/output_folder'
log_dir = os.path.join(output_dir, 'Logs')
results_dir = os.path.join(output_dir, 'Outputs')

# Create output directories if they don't exist
os.makedirs(log_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

# Configuration for AutoDock Vina
center_x, center_y, center_z = 0, 25, 35
size_x, size_y, size_z = 15, 15, 15
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
results = []
for pdbqt_file in os.listdir(pdbqt_dir):
    if pdbqt_file.endswith('.pdbqt'):
        receptor_pdbqt_path = os.path.join(pdbqt_dir, pdbqt_file)
        output_pdbqt = os.path.join(results_dir, pdbqt_file.replace('.pdbqt', '_out.pdbqt'))
        log_file = os.path.join(log_dir, pdbqt_file.replace('.pdbqt', '_log.txt'))
        
        # Prepare the config file for docking
        config_file = os.path.join(output_dir, 'config.txt')
        with open(config_file, 'w') as f:
            f.write(config_template)
            f.write(f"\nreceptor = {receptor_pdbqt_path}\n")

        # Run AutoDock Vina
        vina_cmd = f"vina --receptor {receptor_pdbqt_path} --config {config_file} --log {log_file} --out {output_pdbqt}"
        try:
            subprocess.run(vina_cmd, shell=True, check=True)
            print(f"Docking completed: {output_pdbqt}")

            # Extract binding energy from the log file
            with open(log_file, 'r') as log:
                for line in log:
                    if "REMARK VINA RESULT:" in line:
                        binding_energy = float(re.findall(r"[-+]?\d*\.\d+|\d+", line)[0])
                        results.append({'Receptor': pdbqt_file, 'Binding Energy (kcal/mol)': binding_energy})
                        break
        except subprocess.CalledProcessError as e:
            print(f"Error in docking for {pdbqt_file}: {e}")
            continue

# Save results to a CSV file
df = pd.DataFrame(results)
results_csv = os.path.join(output_dir, 'docking_results.csv')
df.to_csv(results_csv, index=False)
print(f"Results saved to {results_csv}")
