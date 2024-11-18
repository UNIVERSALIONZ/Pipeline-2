import os
import pandas as pd
import re

# Define directories
log_dir = r"C:\Users\NANOPORE\Documents\Nakul\Data\ML\Pipeline_2\DXD_Mutations\Old_Run\Docking\Logs"  # Folder containing log files
output_dir = r"C:\Users\NANOPORE\Documents\Nakul\Data\ML\Pipeline_2\DXD_Mutations\Old_Run\Docking"

# Collect docking results from existing log files
def collect_docking_results():
    results = []
    pdbqt_files = [f for f in os.listdir(log_dir) if f.endswith('_log.txt')]

    for log_file_name in pdbqt_files:
        log_file = os.path.join(log_dir, log_file_name)
        try:
            with open(log_file, 'r') as log:
                lines = log.readlines()
                for line in lines:
                    if re.match(r"\s*1\s+[-+]?[0-9]*\.?[0-9]+", line):
                        # Extracting the binding affinity value for mode 1 only
                        columns = line.split()
                        mode = int(columns[0])
                        binding_energy = float(columns[1])
                        receptor_name = log_file_name.replace('_log.txt', '.pdbqt')
                        results.append({'Receptor': receptor_name, 'Mode': mode, 'Binding Energy (kcal/mol)': binding_energy})
                        break
        except FileNotFoundError:
            print(f"Log file not found for {log_file_name}, docking may have failed.")

    return results

if __name__ == "__main__":
    results = collect_docking_results()

    # Check if results list is empty
    if not results:
        print("No docking results found. Please check if the docking ran successfully and the log files were generated correctly.")
    else:
        # Save results to a CSV file
        df = pd.DataFrame(results)
        results_csv = os.path.join(output_dir, 'docking_results.csv')
        df.to_csv(results_csv, index=False)
        print(f"Results saved to {results_csv}")
