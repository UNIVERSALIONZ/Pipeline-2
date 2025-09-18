"""
Test Suite for Binding Affinity Prediction Pipeline
==================================================

This script tests the core functionality of the binding affinity prediction system
to ensure everything works correctly before running on real data.

Run with: python test_pipeline.py
"""

import sys
import os
import tempfile
import torch
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all required imports work correctly."""
    print("🧪 Testing imports...")
    
    try:
        from binding_affinity_predictor import (
            BindingAffinityPredictor, 
            MultiHeadAttentionTransformer,
            BindingAffinityDataset,
            PositionalEncoding,
            set_random_seeds
        )
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_random_seeds():
    """Test random seed setting functionality."""
    print("🧪 Testing random seed setting...")
    
    try:
        from binding_affinity_predictor import set_random_seeds
        
        # Test that seeds can be set without errors
        set_random_seeds(42)
        
        # Test reproducibility
        set_random_seeds(42)
        x1 = torch.randn(5, 10)
        
        set_random_seeds(42)
        x2 = torch.randn(5, 10)
        
        if torch.allclose(x1, x2):
            print("✅ Random seed setting works correctly")
            return True
        else:
            print("❌ Random seed setting not working")
            return False
            
    except Exception as e:
        print(f"❌ Random seed test failed: {e}")
        return False

def test_model_architecture():
    """Test model architecture creation and forward pass."""
    print("🧪 Testing model architecture...")
    
    try:
        from binding_affinity_predictor import MultiHeadAttentionTransformer
        
        # Create model with small dimensions for testing
        model = MultiHeadAttentionTransformer(
            protein_dim=100,
            ligand_dim=50,
            d_model=64,
            n_heads=4,
            n_layers=2,
            dropout=0.1
        )
        
        # Test forward pass
        batch_size = 8
        protein_emb = torch.randn(batch_size, 100)
        ligand_emb = torch.randn(batch_size, 50)
        
        with torch.no_grad():
            output = model(protein_emb, ligand_emb)
        
        if output.shape == (batch_size,):
            print("✅ Model architecture test passed")
            return True
        else:
            print(f"❌ Model output shape incorrect: {output.shape}")
            return False
            
    except Exception as e:
        print(f"❌ Model architecture test failed: {e}")
        return False

def test_dataset_creation():
    """Test dataset creation and data loading."""
    print("🧪 Testing dataset creation...")
    
    try:
        from binding_affinity_predictor import BindingAffinityDataset
        
        # Create dummy data
        protein_embeddings = torch.randn(10, 3, 100)  # 10 samples, 3 residues, 100-dim
        ligand_embeddings = torch.randn(10, 50)       # 10 samples, 50-dim
        affinities = torch.randn(10)                  # 10 binding affinities
        
        # Create dataset
        dataset = BindingAffinityDataset(protein_embeddings, ligand_embeddings, affinities)
        
        # Test dataset length
        if len(dataset) != 10:
            print(f"❌ Dataset length incorrect: {len(dataset)}")
            return False
        
        # Test data retrieval
        features, affinity = dataset[0]
        expected_feature_dim = 3 * 100 + 50  # flattened protein + ligand
        
        if features.shape[0] == expected_feature_dim:
            print("✅ Dataset creation test passed")
            return True
        else:
            print(f"❌ Feature dimension incorrect: {features.shape[0]}")
            return False
            
    except Exception as e:
        print(f"❌ Dataset creation test failed: {e}")
        return False

class MockESMModel:
    """Mock ESM model for testing without downloading weights."""
    def __init__(self):
        self.eval = MagicMock()
        
    def __call__(self, tokens, repr_layers, return_contacts):
        batch_size, seq_len = tokens.shape
        # Return mock embeddings
        return {
            "representations": {
                33: torch.randn(batch_size, seq_len, 1280)
            }
        }

class MockChemBERTaModel:
    """Mock ChemBERTa model for testing."""
    def __init__(self):
        self.eval = MagicMock()
        
    def __call__(self, **inputs):
        batch_size = inputs['input_ids'].shape[0]
        seq_len = inputs['input_ids'].shape[1]
        return MagicMock(last_hidden_state=torch.randn(batch_size, seq_len, 768))

def test_embedding_extractors():
    """Test protein and ligand embedding extraction (mocked)."""
    print("🧪 Testing embedding extractors (mocked)...")
    
    try:
        # Mock the ESM and transformers imports
        with patch('esm.pretrained.esm2_t33_650M_UR50D') as mock_esm, \
             patch('transformers.AutoTokenizer.from_pretrained') as mock_tokenizer, \
             patch('transformers.AutoModel.from_pretrained') as mock_model:
            
            # Setup mocks
            mock_alphabet = MagicMock()
            mock_alphabet.get_batch_converter.return_value = lambda x: (
                ["protein"], [x[0][1]], torch.randint(0, 20, (1, len(x[0][1]) + 2))
            )
            mock_esm.return_value = (MockESMModel(), mock_alphabet)
            mock_tokenizer.return_value = MagicMock()
            mock_model.return_value = MockChemBERTaModel()
            
            from binding_affinity_predictor import ProteinEmbeddingExtractor, LigandEmbeddingExtractor
            
            # Test protein embedding extractor
            protein_extractor = ProteinEmbeddingExtractor()
            sequences = ["MKVLWAALL"]
            positions = [[0, 2, 4]]
            
            protein_embeddings = protein_extractor.extract_residue_embeddings(sequences, positions)
            
            # Test ligand embedding extractor
            ligand_extractor = LigandEmbeddingExtractor()
            smiles = ["CCO"]
            
            # Mock tokenizer call
            mock_tokenizer_instance = mock_tokenizer.return_value
            mock_tokenizer_instance.return_value = {
                'input_ids': torch.randint(0, 1000, (1, 10)),
                'attention_mask': torch.ones(1, 10)
            }
            
            ligand_embeddings = ligand_extractor.extract_embeddings(smiles)
            
            print("✅ Embedding extractors test passed (mocked)")
            return True
            
    except Exception as e:
        print(f"❌ Embedding extractors test failed: {e}")
        return False

def test_full_pipeline_mock():
    """Test the full pipeline with mocked components."""
    print("🧪 Testing full pipeline (mocked)...")
    
    try:
        with patch('esm.pretrained.esm2_t33_650M_UR50D') as mock_esm, \
             patch('transformers.AutoTokenizer.from_pretrained') as mock_tokenizer, \
             patch('transformers.AutoModel.from_pretrained') as mock_model:
            
            # Setup mocks
            mock_alphabet = MagicMock()
            mock_alphabet.get_batch_converter.return_value = lambda x: (
                ["protein"], [x[0][1]], torch.randint(0, 20, (1, len(x[0][1]) + 2))
            )
            mock_esm.return_value = (MockESMModel(), mock_alphabet)
            mock_tokenizer.return_value = MagicMock()
            mock_model.return_value = MockChemBERTaModel()
            
            # Mock tokenizer instance
            mock_tokenizer_instance = mock_tokenizer.return_value
            mock_tokenizer_instance.return_value = {
                'input_ids': torch.randint(0, 1000, (1, 10)),
                'attention_mask': torch.ones(1, 10)
            }
            
            from binding_affinity_predictor import BindingAffinityPredictor
            
            # Create small config for fast testing
            config = {
                "d_model": 64,
                "n_heads": 4,
                "n_layers": 2,
                "dropout": 0.1,
                "batch_size": 4,
                "learning_rate": 1e-3,
                "epochs": 2,
                "patience": 5
            }
            
            # Create test data
            train_data = {
                "sequences": ["MKVLWAALL", "MKVLWAALL"],
                "residue_positions": [[0, 2, 4], [0, 2, 4]],
                "ligands": ["CCO", "CCC"],
                "affinities": [-6.5, -4.2]
            }
            
            pred_data = {
                "sequences": ["MKVLWAALL"],
                "residue_positions": [[0, 2, 4]],
                "ligands": ["CCN"]
            }
            
            # Initialize predictor
            predictor = BindingAffinityPredictor(config)
            
            # Train model (quick test)
            predictor.train(train_data)
            
            # Make predictions
            predictions = predictor.predict(pred_data)
            
            # Test ranking
            mutant_ids = ["test_mutant"]
            with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
                results_df = predictor.rank_mutants(predictions, mutant_ids, f.name)
                os.unlink(f.name)  # Clean up
            
            if len(results_df) == 1 and "Predicted_Binding_Affinity_kcal_mol" in results_df.columns:
                print("✅ Full pipeline test passed (mocked)")
                return True
            else:
                print("❌ Full pipeline test failed - incorrect output format")
                return False
                
    except Exception as e:
        print(f"❌ Full pipeline test failed: {e}")
        return False

def test_model_save_load():
    """Test model saving and loading functionality."""
    print("🧪 Testing model save/load...")
    
    try:
        from binding_affinity_predictor import MultiHeadAttentionTransformer
        
        # Create model
        model = MultiHeadAttentionTransformer(
            protein_dim=100,
            ligand_dim=50,
            d_model=64,
            n_heads=4,
            n_layers=2
        )
        
        # Create dummy config and scaler
        config = {"protein_dim": 100, "ligand_dim": 50}
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaler.fit([[1], [2], [3]])  # Dummy fit
        
        # Save model
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': config,
                'scaler': scaler,
                'training_history': {"train_loss": [], "val_loss": [], "val_metrics": []}
            }, f.name)
            
            # Load model
            checkpoint = torch.load(f.name, map_location='cpu')
            
            # Verify contents
            if all(key in checkpoint for key in ['model_state_dict', 'config', 'scaler', 'training_history']):
                print("✅ Model save/load test passed")
                os.unlink(f.name)  # Clean up
                return True
            else:
                print("❌ Model save/load test failed - missing keys")
                os.unlink(f.name)
                return False
                
    except Exception as e:
        print(f"❌ Model save/load test failed: {e}")
        return False

def run_all_tests():
    """Run all tests and report results."""
    print("🚀 Starting Binding Affinity Prediction Pipeline Tests")
    print("=" * 60)
    
    tests = [
        ("Imports", test_imports),
        ("Random Seeds", test_random_seeds),
        ("Model Architecture", test_model_architecture),
        ("Dataset Creation", test_dataset_creation),
        ("Embedding Extractors", test_embedding_extractors),
        ("Full Pipeline", test_full_pipeline_mock),
        ("Model Save/Load", test_model_save_load)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name} test...")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Results Summary:")
    print("-" * 30)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<20} {status}")
        if result:
            passed += 1
    
    print("-" * 30)
    print(f"Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The pipeline is ready to use.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)