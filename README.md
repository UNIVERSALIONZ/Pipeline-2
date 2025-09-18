# Deep Learning Binding Affinity Prediction for HAS Mutants

## 🧬 Advanced Transformer-Based Pipeline for Protein-Ligand Binding Prediction

A **publication-ready Python codebase** for predicting binding affinity energies of Class II hyaluronan synthase (HAS) mutants interacting with different sugar ligands using cutting-edge transformer architectures.

### 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run demonstration
python demo_simple.py

# Run with your data
python example_usage.py
```

### ✨ Key Features

- 🧠 **ESM-2 Protein Embeddings**: State-of-the-art protein language model (650M parameters)
- 🧪 **ChemBERTa Molecular Embeddings**: Advanced chemical representation learning
- 🤖 **Multi-Head Attention**: Transformer architecture optimized for binding affinity regression
- 📊 **Publication Ready**: Generates ranked results for experimental validation
- 🔬 **Reproducible**: Complete pipeline with seed management and model checkpointing

### 📁 Main Files

| File | Description |
|------|-------------|
| `binding_affinity_predictor.py` | Main comprehensive pipeline (33KB) |
| `demo_simple.py` | Working demonstration with mock embeddings |
| `example_usage.py` | Complete usage example with real data structure |
| `test_pipeline.py` | Comprehensive test suite |
| `DOCUMENTATION.md` | Complete technical documentation |

### 🎯 Use Case

Predict binding affinities for HAS mutants with different sugar ligands:
- **Training**: Mutants with substrates A (actual) and B (non-substrate)
- **Prediction**: All mutants with ligand C for experimental validation
- **Output**: Ranked list of candidates with confidence scores

### 📊 Example Output

```csv
Mutant_ID,Predicted_Binding_Affinity_kcal_mol,Rank,Confidence_Category
HAS_A48V_D50E_D108A,-8.25,1,Very Strong
HAS_A48L_D50K_D108F,-7.89,2,Strong
HAS_A48I_D50R_D108Y,-6.34,3,Strong
```

### 🔬 Experimental Validation

Top candidates are ready for:
- Isothermal Titration Calorimetry (ITC)
- Surface Plasmon Resonance (SPR)
- Enzymatic activity assays

For detailed documentation, see [DOCUMENTATION.md](DOCUMENTATION.md)

---

**Citation**: If you use this code in your research, please cite our work.

**License**: MIT - see LICENSE file for details.
