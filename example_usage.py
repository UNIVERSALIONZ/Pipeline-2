#!/usr/bin/env python3
"""
Example Usage Script for HAS Binding Affinity Prediction

This script demonstrates how to use the binding affinity prediction pipeline
with real data formats and provides utilities for data preprocessing.

Author: Generated for HAS mutant analysis research
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
from pathlib import Path

# Import the main prediction pipeline
from NN import AffinityPredictor, set_random_seeds, create_example_data

def load_has_mutant_data(data_path: str) -> pd.DataFrame:
    """
    Load HAS mutant data from CSV file.
    
    Expected CSV format:
    - mutant_id: Unique identifier for each mutant (e.g., "A48G_L50V_H108R")
    - residue_sequences: Three active site residue sequences 
    - ligand_type: 'A' (substrate), 'B' (non-substrate), 'C' (prediction target)
    - ligand_smiles: SMILES representation of the ligand
    - binding_affinity: Experimental binding affinity in kcal/mol (for training data)
    
    Args:
        data_path: Path to the CSV data file
        
    Returns:
        Formatted DataFrame ready for the prediction pipeline
    """
    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}")
        print("Using example data instead...")
        return create_example_data()
    
    try:
        df = pd.read_csv(data_path)
        required_cols = ['mutant_id', 'residue_sequences', 'ligand_type', 'ligand_smiles', 'binding_affinity']
        
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Missing required columns. Expected: {required_cols}")
        
        print(f"Loaded {len(df)} samples from {data_path}")
        return df
        
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Using example data instead...")
        return create_example_data()

def process_residue_sequences(residue_string: str) -> list:
    """
    Process residue sequence string into list format.
    
    Handles various input formats:
    - "RES1,RES2,RES3" -> ["RES1", "RES2", "RES3"]  
    - "RES1;RES2;RES3" -> ["RES1", "RES2", "RES3"]
    - "[RES1, RES2, RES3]" -> ["RES1", "RES2", "RES3"]
    
    Args:
        residue_string: String representation of residue sequences
        
    Returns:
        List of three residue sequences
    """
    # Remove brackets and quotes
    clean_string = residue_string.strip('[]"\'')
    
    # Split by comma or semicolon
    if ',' in clean_string:
        sequences = [seq.strip().strip('"\' ') for seq in clean_string.split(',')]
    elif ';' in clean_string:
        sequences = [seq.strip().strip('"\' ') for seq in clean_string.split(';')]
    else:
        # Assume single sequence, split into three equal parts
        sequences = [clean_string]
    
    # Ensure we have exactly 3 sequences
    if len(sequences) == 1:
        # Split single sequence into three parts
        seq = sequences[0]
        part_len = len(seq) // 3
        sequences = [seq[i*part_len:(i+1)*part_len] for i in range(3)]
    elif len(sequences) != 3:
        raise ValueError(f"Expected 3 residue sequences, got {len(sequences)}")
    
    return sequences

def validate_smiles(smiles: str) -> bool:
    """
    Basic SMILES validation.
    
    Args:
        smiles: SMILES string to validate
        
    Returns:
        True if SMILES appears valid, False otherwise
    """
    if not isinstance(smiles, str) or len(smiles) == 0:
        return False
    
    # Very basic validation - just check for reasonable characters
    valid_chars = set('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789()[]{}=+-.#@:%\\/')
    smiles_chars = set(smiles)
    
    # Check if all characters are in the valid set
    if not smiles_chars.issubset(valid_chars):
        return False
    
    # Check balanced parentheses
    paren_count = smiles.count('(') - smiles.count(')')
    bracket_count = smiles.count('[') - smiles.count(']')
    
    return paren_count == 0 and bracket_count == 0

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the input data for the prediction pipeline.
    
    Args:
        df: Raw input DataFrame
        
    Returns:
        Preprocessed DataFrame ready for modeling
    """
    processed_df = df.copy()
    
    # Process residue sequences
    if isinstance(processed_df['residue_sequences'].iloc[0], str):
        processed_df['residue_sequences'] = processed_df['residue_sequences'].apply(process_residue_sequences)
    
    # Validate SMILES
    valid_smiles = processed_df['ligand_smiles'].apply(validate_smiles)
    if not valid_smiles.all():
        print(f"Warning: {(~valid_smiles).sum()} invalid SMILES found")
    
    # Remove samples with invalid data
    processed_df = processed_df[valid_smiles].reset_index(drop=True)
    
    # Sort by mutant_id and ligand_type for consistent processing
    processed_df = processed_df.sort_values(['mutant_id', 'ligand_type']).reset_index(drop=True)
    
    print(f"Preprocessed data: {len(processed_df)} valid samples")
    return processed_df

def run_quick_validation(predictor: AffinityPredictor, sample_data: pd.DataFrame):
    """
    Run a quick validation of the pipeline with a small subset of data.
    
    Args:
        predictor: Initialized AffinityPredictor instance
        sample_data: Sample data for validation
    """
    print("\n" + "="*50)
    print("QUICK VALIDATION RUN")
    print("="*50)
    
    # Use small subset for quick validation
    small_data = sample_data.head(20).copy()
    
    try:
        # Prepare data
        train_loader, val_loader, test_data = predictor.prepare_data(small_data)
        
        # Create model
        model = predictor.create_model(hidden_dim=64, num_layers=1)  # Smaller for quick test
        
        # Quick training (3 epochs)
        history = predictor.train_model(train_loader, val_loader, num_epochs=3)
        
        # Generate predictions
        if len(test_data) > 0:
            predictions = predictor.predict_affinities(test_data)
            print(f"Generated {len(predictions)} predictions successfully")
            print("\nSample predictions:")
            print(predictions.head().to_string(index=False))
        else:
            print("No test data available for prediction")
        
        print("\nQuick validation completed successfully!")
        
    except Exception as e:
        print(f"Validation failed: {e}")
        import traceback
        traceback.print_exc()

def create_data_template(output_path: str = "data_template.csv"):
    """
    Create a template CSV file showing the expected data format.
    
    Args:
        output_path: Path where to save the template file
    """
    template_data = {
        'mutant_id': [
            'A48G_L50V_H108R',
            'A48G_L50V_H108R', 
            'A48G_L50V_H108R',
            'A48D_L50I_H108K',
            'A48D_L50I_H108K',
            'A48D_L50I_H108K'
        ],
        'residue_sequences': [
            '["GLYALA", "VALTRP", "ARGPHE"]',
            '["GLYALA", "VALTRP", "ARGPHE"]',
            '["GLYALA", "VALTRP", "ARGPHE"]',
            '["ASPALA", "ILETRP", "LYSPHE"]',
            '["ASPALA", "ILETRP", "LYSPHE"]', 
            '["ASPALA", "ILETRP", "LYSPHE"]'
        ],
        'ligand_type': ['A', 'B', 'C', 'A', 'B', 'C'],
        'ligand_smiles': [
            'CC(C)(C)OC(=O)N[C@@H](CO)C(=O)O',  # Substrate A
            'C[C@H]1O[C@H](O)[C@H](O)[C@@H](O)[C@@H]1O',  # Non-substrate B
            'O[C@H]1[C@H](O)[C@@H](O)[C@H](O)[C@@H]1O',    # Non-substrate C (prediction target)
            'CC(C)(C)OC(=O)N[C@@H](CO)C(=O)O',  # Substrate A
            'C[C@H]1O[C@H](O)[C@H](O)[C@@H](O)[C@@H]1O',  # Non-substrate B
            'O[C@H]1[C@H](O)[C@@H](O)[C@H](O)[C@@H]1O'     # Non-substrate C (prediction target)
        ],
        'binding_affinity': [-8.5, -6.2, np.nan, -7.8, -5.9, np.nan]  # NaN for prediction targets
    }
    
    template_df = pd.DataFrame(template_data)
    template_df.to_csv(output_path, index=False)
    
    print(f"\nData template saved to: {output_path}")
    print("\nTemplate format:")
    print(template_df.to_string(index=False))
    print(f"\nInstructions:")
    print("1. Replace example mutant_ids with your actual HAS mutant identifiers")
    print("2. Update residue_sequences with the three active site residue sequences for each mutant")
    print("3. Provide SMILES strings for your ligands A (substrate), B (non-substrate), C (prediction target)")
    print("4. Fill in experimental binding_affinity values for training data (ligands A and B)")
    print("5. Leave binding_affinity as NaN for prediction targets (ligand C)")

def main():
    """
    Main execution function with command-line interface.
    """
    print("HAS Binding Affinity Prediction Pipeline")
    print("="*50)
    
    # Set reproducibility
    set_random_seeds(42)
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='HAS Binding Affinity Prediction')
    parser.add_argument('--data', type=str, help='Path to input CSV data file')
    parser.add_argument('--template', action='store_true', help='Create data template file')
    parser.add_argument('--validate', action='store_true', help='Run quick validation')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--output', type=str, default='predictions.csv', help='Output predictions file')
    
    args = parser.parse_args()
    
    # Create template if requested
    if args.template:
        create_data_template()
        return
    
    # Load and preprocess data
    if args.data:
        raw_data = load_has_mutant_data(args.data)
    else:
        print("No data file specified, using example data...")
        raw_data = create_example_data()
    
    data = preprocess_data(raw_data)
    
    # Initialize predictor
    predictor = AffinityPredictor()
    
    # Run validation if requested
    if args.validate:
        run_quick_validation(predictor, data)
        return
    
    # Full pipeline execution
    print(f"\nRunning full pipeline with {len(data)} samples...")
    
    try:
        # Prepare datasets
        train_loader, val_loader, test_data = predictor.prepare_data(data)
        
        # Create and train model
        model = predictor.create_model()
        history = predictor.train_model(train_loader, val_loader, num_epochs=args.epochs)
        
        # Generate predictions
        if len(test_data) > 0:
            predictions = predictor.predict_affinities(test_data)
            
            # Save results
            predictions.to_csv(args.output, index=False)
            print(f"\nPredictions saved to: {args.output}")
            
            # Display summary
            print(f"\nSUMMARY:")
            print(f"- Processed {len(data)} total samples")
            print(f"- Generated predictions for {len(predictions)} mutants with ligand C")
            print(f"- Top predicted binder: {predictions.iloc[0]['mutant_id']} "
                  f"(ΔG = {predictions.iloc[0]['predicted_binding_affinity']:.2f} kcal/mol)")
            print(f"- Prediction range: {predictions['predicted_binding_affinity'].min():.2f} to "
                  f"{predictions['predicted_binding_affinity'].max():.2f} kcal/mol")
            
            # Save model
            model_path = args.output.replace('.csv', '_model.pth')
            predictor.save_model(model_path)
            print(f"- Model saved to: {model_path}")
            
        else:
            print("No test data available for prediction (no ligand C samples found)")
            
    except Exception as e:
        print(f"Pipeline execution failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("\nPipeline completed successfully!")
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)