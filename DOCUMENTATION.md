# Deep Learning Binding Affinity Prediction for HAS Mutants

## Overview

This repository provides a **publication-ready Python codebase** for predicting binding affinity energies of Class II hyaluronan synthase (HAS) mutants interacting with different sugar ligands using state-of-the-art transformer architectures.

### Key Features

- 🧬 **ESM-2 Protein Embeddings**: Uses Meta AI's ESM-2 transformer for high-quality protein representations
- 🧪 **ChemBERTa Molecular Embeddings**: Leverages transformer-based molecular representations for ligands
- 🤖 **Multi-Head Attention Architecture**: Implements advanced transformer models optimized for regression tasks
- 📊 **Reproducible Pipeline**: Complete end-to-end workflow with reproducibility guarantees
- 🎯 **Publication Ready**: Generates ranked results suitable for experimental validation

## Scientific Background

### Problem Statement

Class II hyaluronan synthase (HAS) enzymes catalyze the synthesis of hyaluronic acid, a critical component of the extracellular matrix. Understanding how mutations in the active site affect substrate binding is crucial for:

- Drug design and therapeutic interventions
- Understanding enzyme specificity and evolution
- Optimizing enzyme variants for biotechnological applications

### Computational Approach

This pipeline addresses the binding affinity prediction problem using:

1. **Protein Representation**: ESM-2 embeddings capture evolutionary and structural information from protein sequences
2. **Ligand Representation**: ChemBERTa embeddings encode molecular properties from SMILES strings
3. **Fusion Architecture**: Multi-head attention transformers learn complex protein-ligand interaction patterns
4. **Regression Task**: Direct prediction of binding affinity energies (ΔG in kcal/mol)

## Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- 16GB+ RAM

### Dependencies

Install all required packages:

```bash
pip install -r requirements.txt
```

### Additional Setup

For ESM-2 protein embeddings:
```bash
pip install fair-esm
```

For molecular processing:
```bash
# Install RDKit
conda install -c conda-forge rdkit

# Or using pip
pip install rdkit
```

## Usage

### Quick Start

1. **Prepare your data** in the required format (see Data Format section)
2. **Configure the pipeline** by editing `config.yaml`
3. **Run the example script**:

```bash
python example_usage.py
```

### Custom Data Usage

Replace the sample data in `example_usage.py` with your experimental data:

```python
def load_experimental_data():
    # Load your protein sequences
    protein_sequences = ["MKVLWAALL..."]  # Full protein sequences
    
    # Define active site positions (0-indexed)
    residue_positions = [[47, 49, 107]]  # Positions 48, 50, 108
    
    # Define ligand SMILES
    ligands = {
        "A": "C1C(C(C(C(O1)CO)O)O)O",  # Actual substrate
        "B": "C1C(C(C(C(O1)CO)O)O)N",  # Non-substrate (training)
        "C": "C1C(C(C(C(O1)CO)O)O)S"   # Prediction target
    }
    
    # Binding affinities from docking/experiments
    binding_affinities = {
        ("mutant_1", "A"): -6.5,  # kcal/mol
        ("mutant_1", "B"): -4.2,
        # Add more data...
    }
    
    return {...}
```

### Direct API Usage

```python
from binding_affinity_predictor import BindingAffinityPredictor, set_random_seeds

# Set reproducibility
set_random_seeds(42)

# Configure model
config = {
    "d_model": 512,
    "n_heads": 8,
    "n_layers": 6,
    "dropout": 0.1,
    "batch_size": 16,
    "learning_rate": 1e-4,
    "epochs": 100
}

# Initialize predictor
predictor = BindingAffinityPredictor(config)

# Prepare training data
train_data = {
    "sequences": ["MKVLWAALL..."],
    "residue_positions": [[47, 49, 107]],
    "ligands": ["C1C(C(C(C(O1)CO)O)O)O"],
    "affinities": [-6.5]
}

# Train model
predictor.train(train_data)

# Make predictions
pred_data = {
    "sequences": ["MKVLWAALL..."],
    "residue_positions": [[47, 49, 107]], 
    "ligands": ["C1C(C(C(C(O1)CO)O)O)S"]
}

predictions = predictor.predict(pred_data)
results = predictor.rank_mutants(predictions, ["mutant_1"])
```

## Data Format

### Input Requirements

#### 1. Protein Sequences
- **Format**: Full amino acid sequences (single letter code)
- **Length**: Variable length supported
- **Example**: `"MKVLWAALLVTFLAGCQAKVEQAVETEPEPELR..."`

#### 2. Active Site Residues
- **Format**: List of residue positions for each mutant
- **Indexing**: 0-based indexing
- **Count**: Exactly 3 residues per mutant
- **Example**: `[[47, 49, 107], [47, 49, 107]]`

#### 3. Ligand SMILES
- **Format**: Valid SMILES strings
- **Types**: Three ligand types (A, B, C)
- **Example**: 
  - A (substrate): `"C1C(C(C(C(O1)CO)O)O)O"`
  - B (non-substrate): `"C1C(C(C(C(O1)CO)O)O)N"`
  - C (prediction target): `"C1C(C(C(C(O1)CO)O)O)S"`

#### 4. Binding Affinities
- **Format**: Numeric values in kcal/mol
- **Range**: Typically -15 to 0 kcal/mol
- **Training**: Required for mutants with ligands A and B
- **Prediction**: Not required for ligand C

### Output Format

#### Binding Affinity Rankings CSV

```csv
Mutant_ID,Predicted_Binding_Affinity_kcal_mol,Rank,Confidence_Category
HAS_A48V_D50E_D108A,-8.25,1,Very Strong
HAS_A48L_D50K_D108F,-7.89,2,Strong
HAS_A48I_D50R_D108Y,-6.34,3,Strong
...
```

## Model Architecture

### Core Components

1. **ESM-2 Protein Encoder**
   - Pre-trained transformer language model
   - 650M parameters
   - Extracts residue-level embeddings (1280-dim)

2. **ChemBERTa Molecular Encoder**
   - Pre-trained on ZINC database
   - Processes SMILES strings
   - Generates molecular embeddings (768-dim)

3. **Multi-Head Attention Fusion**
   - 8 attention heads
   - 6 transformer layers
   - Cross-attention between protein and ligand features

4. **Regression Head**
   - Multi-layer perceptron
   - Layer normalization and dropout
   - Outputs binding affinity (kcal/mol)

### Design Rationale

- **Transformer Architecture**: Captures long-range dependencies and complex interactions
- **Pre-trained Embeddings**: Leverages large-scale pre-training for better generalization
- **Cross-Attention**: Explicitly models protein-ligand interactions
- **Regularization**: Dropout and weight decay prevent overfitting

## Training Strategy

### Data Splitting
- **Training**: Mutants with ligands A and B
- **Validation**: 20% of training data
- **Prediction**: All mutants with ligand C

### Optimization
- **Optimizer**: AdamW with weight decay
- **Learning Rate**: 1e-4 with plateau scheduling
- **Batch Size**: 16 (adjustable based on GPU memory)
- **Early Stopping**: Patience of 15 epochs

### Regularization
- **Dropout**: 0.1 in all layers
- **Weight Decay**: 1e-5
- **Gradient Clipping**: Max norm of 1.0
- **Layer Normalization**: Applied throughout

## Validation and Metrics

### Performance Metrics
- **MSE**: Mean Squared Error
- **MAE**: Mean Absolute Error  
- **R²**: Coefficient of determination
- **Pearson r**: Correlation coefficient

### Cross-Validation
- Time series split for temporal validation
- Leave-one-out for small datasets
- Stratified splits for balanced evaluation

## Reproducibility

### Random Seed Management
```python
set_random_seeds(42)  # Sets all framework seeds
```

### Model Checkpointing
- Automatic saving of best model
- Complete state preservation
- Configuration storage

### Environment Specification
- Fixed package versions in `requirements.txt`
- Docker support (optional)
- Conda environment export

## Experimental Validation Guidelines

### Top Candidate Selection
1. **Binding Affinity**: Lower ΔG indicates stronger binding
2. **Confidence Categories**:
   - Very Strong: ΔG < -8 kcal/mol
   - Strong: -8 ≤ ΔG < -6 kcal/mol
   - Moderate: -6 ≤ ΔG < -4 kcal/mol
   - Weak: ΔG ≥ -4 kcal/mol

### Recommended Experiments
1. **Isothermal Titration Calorimetry (ITC)**
   - Direct binding affinity measurement
   - Thermodynamic parameters
   
2. **Surface Plasmon Resonance (SPR)**
   - Real-time binding kinetics
   - Association/dissociation rates

3. **Enzymatic Activity Assays**
   - Functional validation
   - Substrate specificity

## Advanced Features

### Custom Loss Functions
```python
# Implement custom loss for your specific use case
class RankingLoss(nn.Module):
    def forward(self, predictions, targets):
        # Custom ranking loss implementation
        pass
```

### Ensemble Predictions
```python
# Train multiple models for ensemble predictions
ensemble_predictions = []
for seed in [42, 123, 456]:
    predictor = BindingAffinityPredictor(config)
    predictor.train(train_data)
    pred = predictor.predict(test_data)
    ensemble_predictions.append(pred)

final_predictions = np.mean(ensemble_predictions, axis=0)
```

### Uncertainty Quantification
- Monte Carlo dropout
- Bootstrap aggregating
- Bayesian neural networks

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size
   - Use gradient accumulation
   - Enable mixed precision training

2. **Slow Training**
   - Use smaller model (reduce layers/heads)
   - Optimize data loading
   - Enable GPU acceleration

3. **Poor Performance**
   - Increase training data
   - Adjust hyperparameters
   - Check data quality

### Performance Optimization

```python
# Enable mixed precision training
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    predictions = model(features)
    loss = criterion(predictions, targets)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

## Citation

If you use this code in your research, please cite:

```bibtex
@software{has_binding_affinity_prediction,
  title={Deep Learning Binding Affinity Prediction for HAS Mutants},
  author={Automated ML Pipeline},
  year={2024},
  url={https://github.com/UNIVERSALIONZ/Pipeline-2}
}
```

## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Support

For questions and support:
- Open an issue on GitHub
- Check the troubleshooting section
- Review the example usage

## Changelog

### v1.0.0 (2024)
- Initial release
- ESM-2 and ChemBERTa integration
- Multi-head attention transformer
- Complete training pipeline
- Publication-ready outputs