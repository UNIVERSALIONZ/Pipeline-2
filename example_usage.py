"""
Example Usage Script for Binding Affinity Prediction
====================================================

This script demonstrates how to use the binding affinity prediction pipeline
with your own data. Replace the sample data with your actual experimental data.

Requirements:
- Protein sequences for each mutant
- Active site residue positions (3 residues per mutant)
- SMILES strings for ligands A, B, and C
- Binding affinity data for training (mutants with ligands A and B)

Usage:
    python example_usage.py
"""

import os
import yaml
import pandas as pd
import numpy as np
from binding_affinity_predictor import BindingAffinityPredictor, set_random_seeds

def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_experimental_data():
    """
    Load your experimental data here.
    
    Replace this function with code to load your actual data.
    This should return dictionaries in the format expected by the predictor.
    """
    
    # Example data structure - replace with your data loading code
    
    # Protein sequences for each mutant (full sequences)
    protein_sequences = [
        "MKVLWAALLVTFLAGCQAKVEQAVETEPEPELRQQTEWQSGQRWELALGRAPHQVPVQLQRVAAHRDGRVSVAQLRFNASLHRRWPRTSFGPPEKRFDDDFYELLFSRPLHRLADFSSKHLFQRLAELKAELQAEVFIRFGDRPPETAEAIEKVLQAVVQAVKKAGGPPGPPPPPAALATPDEAAAASAAASAAATPPATPPATPPPAAATPPATPPPAAATPPATPPPAAATPPATPPPAAATPPATPPPAAATPPATPPPAAATPPATPPPAAATPPATPPPAAATPPATPPPAAATPPATPPPAAATPPATPPPAAATPPATPPPAAATPPATPPPAAATPPATPPPAAATPPATPPPAAATPPATPPPAAATPPATPPPAAATPPATPPPAAATPPATPPPAAATPPATPPPAAATPPATPPPAAATPPATPPPAAATPPATPPPAAATPPATPPPAAATPPATPPPAAATPPATPPPAAATPPATPPP",
        # Add more mutant sequences here...
    ]
    
    # Active site residue positions for each mutant (0-indexed)
    residue_positions = [
        [47, 49, 107],  # Positions 48, 50, 108 in 1-indexed notation
        # Add corresponding positions for other mutants...
    ]
    
    # SMILES strings for ligands
    ligands = {
        "A": "C1C(C(C(C(O1)CO)O)O)O",  # Sugar A - actual substrate
        "B": "C1C(C(C(C(O1)CO)O)O)N",  # Sugar B - non-substrate for training
        "C": "C1C(C(C(C(O1)CO)O)O)S"   # Sugar C - prediction target
    }
    
    # Binding affinity data (kcal/mol) - from docking or experimental measurements
    # Training data: mutants with ligands A and B
    binding_affinities = {
        ("mutant_1", "A"): -6.5,
        ("mutant_1", "B"): -4.2,
        # Add more binding affinity data...
    }
    
    # Mutant identifiers
    mutant_ids = [
        "HAS_A48V_D50E_D108A",  # Example mutant ID
        # Add more mutant IDs...
    ]
    
    return {
        "sequences": protein_sequences,
        "positions": residue_positions, 
        "ligands": ligands,
        "affinities": binding_affinities,
        "mutant_ids": mutant_ids
    }

def prepare_training_data(data):
    """Prepare data in the format expected by the predictor."""
    
    sequences = []
    positions = []
    ligands = []
    affinities = []
    
    # Combine data from ligands A and B for training
    for i, (seq, pos) in enumerate(zip(data["sequences"], data["positions"])):
        mutant_id = data["mutant_ids"][i]
        
        # Add data for ligand A
        if (mutant_id, "A") in data["affinities"]:
            sequences.append(seq)
            positions.append(pos)
            ligands.append(data["ligands"]["A"])
            affinities.append(data["affinities"][(mutant_id, "A")])
        
        # Add data for ligand B  
        if (mutant_id, "B") in data["affinities"]:
            sequences.append(seq)
            positions.append(pos)
            ligands.append(data["ligands"]["B"])
            affinities.append(data["affinities"][(mutant_id, "B")])
    
    return {
        "sequences": sequences,
        "residue_positions": positions,
        "ligands": ligands,
        "affinities": affinities
    }

def prepare_prediction_data(data):
    """Prepare data for prediction with ligand C."""
    
    return {
        "sequences": data["sequences"],
        "residue_positions": data["positions"],
        "ligands": [data["ligands"]["C"]] * len(data["sequences"])
    }

def main():
    """Main execution function."""
    
    print("🧬 HAS Mutant Binding Affinity Prediction Pipeline")
    print("=" * 55)
    
    # Load configuration
    config = load_config()
    
    # Set random seed for reproducibility
    set_random_seeds(config["random_seed"])
    
    # Create output directories
    os.makedirs(config["output"]["model_save_path"], exist_ok=True)
    os.makedirs(config["output"]["results_save_path"], exist_ok=True)
    os.makedirs(config["output"]["plots_save_path"], exist_ok=True)
    
    # Load experimental data
    print("📊 Loading experimental data...")
    experimental_data = load_experimental_data()
    
    # Prepare training data (mutants with ligands A and B)
    print("🔧 Preparing training data...")
    train_data = prepare_training_data(experimental_data)
    
    # Split training data for validation
    n_train = int(0.8 * len(train_data["sequences"]))
    val_data = {
        "sequences": train_data["sequences"][n_train:],
        "residue_positions": train_data["residue_positions"][n_train:],
        "ligands": train_data["ligands"][n_train:],
        "affinities": train_data["affinities"][n_train:]
    }
    train_data = {
        "sequences": train_data["sequences"][:n_train],
        "residue_positions": train_data["residue_positions"][:n_train],
        "ligands": train_data["ligands"][:n_train],
        "affinities": train_data["affinities"][:n_train]
    }
    
    print(f"📈 Training samples: {len(train_data['sequences'])}")
    print(f"📊 Validation samples: {len(val_data['sequences'])}")
    
    # Initialize predictor
    print("🤖 Initializing binding affinity predictor...")
    predictor_config = {**config["model"], **config["training"]}
    predictor = BindingAffinityPredictor(predictor_config)
    
    # Train model
    print("🚀 Starting model training...")
    predictor.train(train_data, val_data)
    
    # Prepare prediction data (all mutants with ligand C)
    print("🔮 Preparing prediction data for ligand C...")
    pred_data = prepare_prediction_data(experimental_data)
    
    # Make predictions
    print("🎯 Making binding affinity predictions...")
    predictions = predictor.predict(pred_data)
    
    # Rank mutants by predicted binding affinity
    print("📋 Ranking mutants for experimental validation...")
    results_file = os.path.join(config["output"]["results_save_path"], "binding_affinity_rankings.csv")
    results_df = predictor.rank_mutants(predictions, experimental_data["mutant_ids"], results_file)
    
    # Save training plots
    plot_file = os.path.join(config["output"]["plots_save_path"], "training_history.png")
    predictor.plot_training_history(plot_file)
    
    # Save final model
    model_file = os.path.join(config["output"]["model_save_path"], "final_binding_affinity_model.pth")
    predictor.save_model(model_file)
    
    # Print summary results
    print("\n🎉 Analysis Complete!")
    print("-" * 40)
    print("📊 Top 5 Candidates for Experimental Validation:")
    
    for idx, row in results_df.head().iterrows():
        print(f"  {row['Rank']}. {row['Mutant_ID']}")
        print(f"     Predicted ΔG: {row['Predicted_Binding_Affinity_kcal_mol']:.2f} kcal/mol")
        print(f"     Confidence: {row['Confidence_Category']}")
        print()
    
    print(f"📁 Results saved to: {results_file}")
    print(f"🤖 Model saved to: {model_file}")
    print(f"📈 Training plots saved to: {plot_file}")
    
    print("\n🔬 Ready for Experimental Validation!")
    print("Consider the top-ranked mutants for isothermal titration calorimetry (ITC)")
    print("or other binding affinity measurement techniques.")

if __name__ == "__main__":
    main()