import os

# Directory containing your 400 PDBQT files
input_folder = r"C:\Users\NANOPORE\Documents\Nakul\Data\ML\Pipeline_2\PDBQT"
output_folder = r"C:\Users\NANOPORE\Documents\Nakul\Data\ML\Pipeline_2\PDBQT_Charged"

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Loop through all PDBQT files in the input folder
for pdbqt_file in os.listdir(input_folder):
    if pdbqt_file.endswith('.pdbqt'):
        pdbqt_path = os.path.join(input_folder, pdbqt_file)
        output_pdbqt_path = os.path.join(output_folder, pdbqt_file)

        with open(pdbqt_path, 'r') as file:
            lines = file.readlines()

        # Flag to track if modifications were made
        modifications_made = False

        # Loop through each line to search for Mn ions and modify their charge
        for i, line in enumerate(lines):
            if 'MN' in line and len(line) >= 73:  # Check if the line contains Mn and is long enough
                # Change the charge at column 73 to '2'
                modified_line = line[:72] + '2.' + line[73:]
                lines[i] = modified_line
                modifications_made = True
                print(f"Modified charge of Mn ion in file: {pdbqt_file} on line {i+1}")
	# Loop through each line to search for Mn ions and modify their charge
        for i, line in enumerate(lines):
            if 'MN' in line and len(line) >= 73:  # Check if the line contains Mn and is long enough
                # Change the charge at column 73 to '2'
                modified_line = line[:79] + 'n' + line[80:]
                lines[i] = modified_line
                modifications_made = True
                print(f"Modified charge of Mn ion in file: {pdbqt_file} on line {i+1}")

        # Write the modified lines to a new file in the output folder
        with open(output_pdbqt_path, 'w') as file:
            file.writelines(lines)

        if not modifications_made:
            print(f"No Mn ion found or no modification needed for file: {pdbqt_file}")

print("Modification complete for all PDBQT files.")
