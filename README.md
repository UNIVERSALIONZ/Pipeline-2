# Advanced Deep Learning Binding Affinity Prediction for HAS Mutants

A robust, publication-ready Python pipeline for predicting binding affinity energies of Class II hyaluronan synthase (HAS) mutants with sugar ligands using state-of-the-art transformer architectures.

## 🔬 Overview

This pipeline implements cutting-edge deep learning methods for protein-ligand binding affinity prediction, specifically designed for HAS mutant analysis. The system leverages:

- **ESM-2 Transformer**: State-of-the-art protein embeddings for active site residues
- **ChemBERTa/Chemformer**: Advanced molecular transformers for ligand representation
- **Multi-Head Cross-Attention**: Sophisticated protein-ligand interaction modeling
- **Comprehensive Pipeline**: End-to-end solution from data preprocessing to ranked predictions

## 📊 Key Features

- ✅ **Publication-Ready**: Designed for peer-reviewed manuscript quality
- ✅ **Reproducible Results**: Fixed random seeds and comprehensive logging
- ✅ **Advanced Architecture**: Multi-head attention transformers with residual connections
- ✅ **Easy Implementation**: Master's-level bioengineering student friendly
- ✅ **Experimental Validation**: Ranked output for isothermal titration calorimetry (ITC)
- ✅ **Model Persistence**: Save/load trained models for future use

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/UNIVERSALIONZ/Pipeline-2.git
cd Pipeline-2

# Install dependencies
pip install -r requirements.txt

# For full functionality, install optional molecular packages:
pip install rdkit-pypi fair-esm deepchem
```

### Basic Usage

```python
from NN import AffinityPredictor, create_example_data

# Initialize predictor
predictor = AffinityPredictor()

# Load your data (or use example data)
data = create_example_data()

# Prepare datasets
train_loader, val_loader, test_data = predictor.prepare_data(data)

# Create and train model
model = predictor.create_model()
history = predictor.train_model(train_loader, val_loader, num_epochs=100)

# Generate predictions
predictions = predictor.predict_affinities(test_data)

# Save results
predictions.to_csv('has_predictions.csv', index=False)
```

### Command Line Interface

```bash
# Create data template
python example_usage.py --template

# Run quick validation
python example_usage.py --validate

# Full pipeline with custom data
python example_usage.py --data your_data.csv --epochs 100 --output results.csv
```

## 📁 Data Format

Your input CSV must contain these columns:

| Column | Description | Example |
|--------|-------------|---------|
| `mutant_id` | Unique mutant identifier | "A48G_L50V_H108R" |
| `residue_sequences` | Three active site residue sequences | ["GLYALA", "VALTRP", "ARGPHE"] |
| `ligand_type` | Ligand category | "A" (substrate), "B" (non-substrate), "C" (prediction target) |
| `ligand_smiles` | SMILES representation | "CC(C)(C)OC(=O)N[C@@H](CO)C(=O)O" |
| `binding_affinity` | Experimental affinity (kcal/mol) | -8.5 (NaN for prediction targets) |

### Example Data Structure

```csv
mutant_id,residue_sequences,ligand_type,ligand_smiles,binding_affinity
A48G_L50V_H108R,"[""GLYALA"", ""VALTRP"", ""ARGPHE""]",A,CC(C)(C)OC(=O)N[C@@H](CO)C(=O)O,-8.5
A48G_L50V_H108R,"[""GLYALA"", ""VALTRP"", ""ARGPHE""]",B,C[C@H]1O[C@H](O)[C@H](O)[C@@H](O)[C@@H]1O,-6.2
A48G_L50V_H108R,"[""GLYALA"", ""VALTRP"", ""ARGPHE""]",C,O[C@H]1[C@H](O)[C@@H](O)[C@H](O)[C@@H]1O,
```

## 🏗️ Architecture Details

### Model Components

1. **ESM-2 Protein Embedder**
   - Extracts 1280-dimensional embeddings for each active site residue
   - Uses Facebook's ESM-2 transformer (650M parameters)
   - Handles three critical HAS active site positions

2. **Molecular Transformer Embedder**
   - ChemBERTa/Chemformer for ligand SMILES encoding
   - 768-dimensional molecular representations
   - Captures chemical structure and properties

3. **Multi-Head Cross-Attention Fusion**
   - 8 attention heads for diverse interaction patterns
   - Scaled dot-product attention between protein and ligand
   - Residual connections and layer normalization

4. **Deep Regression Head**
   - Multi-layer architecture with dropout regularization
   - Layer normalization for stable training
   - Single output for binding affinity (ΔG kcal/mol)

### Training Strategy

- **Data Split**: Train on ligands A & B, predict on ligand C
- **Loss Function**: Mean Squared Error (MSE)
- **Optimizer**: AdamW with weight decay
- **Learning Rate**: Adaptive scheduling with ReduceLROnPlateau
- **Regularization**: Dropout, layer normalization, gradient clipping

## 📈 Evaluation Metrics

The pipeline provides comprehensive evaluation:

- **MSE**: Mean Squared Error
- **RMSE**: Root Mean Squared Error  
- **MAE**: Mean Absolute Error
- **R²**: Coefficient of determination

## 🔬 Experimental Validation Workflow

1. **Model Training**: Train on known HAS mutant data (ligands A & B)
2. **Prediction**: Generate ranked predictions for ligand C
3. **Selection**: Choose top-ranked mutants for experimental validation
4. **ITC Validation**: Measure binding affinities using isothermal titration calorimetry
5. **Model Refinement**: Update model with new experimental data

## 📊 Output Format

The pipeline generates ranked predictions in CSV format:

```csv
mutant_id,predicted_binding_affinity,rank
A48G_L50V_H108R,-9.2,1
A48D_L50I_H108K,-8.7,2
A48F_L50Y_H108W,-8.3,3
...
```

Lower predicted affinity values indicate stronger binding (better candidates).

## 🔧 Advanced Configuration

### Model Hyperparameters

```python
model = predictor.create_model(
    protein_dim=1280,    # ESM-2 embedding dimension
    ligand_dim=768,      # ChemBERTa embedding dimension
    hidden_dim=512,      # Internal representation size
    num_heads=8,         # Multi-head attention heads
    num_layers=3,        # Cross-attention layers
    dropout=0.1          # Regularization strength
)
```

### Training Parameters

```python
history = predictor.train_model(
    train_loader, 
    val_loader,
    num_epochs=100,      # Training epochs
    learning_rate=1e-4,  # Initial learning rate
    weight_decay=1e-5    # L2 regularization
)
```

## 🔍 Troubleshooting

### Common Issues

1. **Memory Errors**: Reduce batch size or model dimensions
2. **Installation Issues**: Use conda environment for complex dependencies
3. **Data Format Errors**: Check CSV structure and SMILES validity
4. **Performance Issues**: Enable GPU acceleration with CUDA

### GPU Acceleration

```python
# Enable GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
predictor = AffinityPredictor(device=device)
```

## 📚 Dependencies

### Core Requirements
- PyTorch ≥ 2.0.0
- Transformers ≥ 4.30.0
- NumPy ≥ 1.24.0
- Pandas ≥ 2.0.0
- Scikit-learn ≥ 1.3.0

### Optional (for full functionality)
- fair-esm ≥ 2.0.0 (ESM-2 models)
- rdkit-pypi ≥ 2022.9.5 (molecular handling)
- deepchem ≥ 2.7.0 (ChemBERTa models)

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed for academic use. Please cite appropriately in publications.

## 📞 Support

For questions or issues:
1. Check the troubleshooting section
2. Review example usage scripts
3. Open an issue on GitHub
4. Contact the development team

## 🎯 Citation

```bibtex
@software{has_binding_affinity_predictor,
  title={Advanced Deep Learning Binding Affinity Prediction for HAS Mutants},
  author={Your Research Team},
  year={2024},
  url={https://github.com/UNIVERSALIONZ/Pipeline-2}
}
```

---

**Ready for experimental validation!** This pipeline provides publication-quality binding affinity predictions to guide your HAS mutant design and experimental validation efforts.
