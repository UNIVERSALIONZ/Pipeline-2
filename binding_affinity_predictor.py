"""
Advanced Deep Learning Binding Affinity Prediction for HAS Mutants
==================================================================

A robust, publication-ready Python codebase to predict binding affinity energies 
of Class II hyaluronan synthase (HAS) mutants interacting with different sugar 
ligands using transformer-based deep learning architectures.

Author: Automated ML Pipeline
Date: 2024
License: MIT
"""

import os
import random
import warnings
import logging
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import yaml

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def set_random_seeds(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility across all frameworks.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Random seeds set to {seed} for reproducibility")

class ProteinEmbeddingExtractor:
    """
    Extract protein embeddings using ESM-2 transformer model.
    
    ESM-2 is the state-of-the-art protein language model developed by Meta AI
    that provides high-quality representations for protein sequences.
    """
    
    def __init__(self, model_name: str = "esm2_t33_650M_UR50D"):
        """
        Initialize ESM-2 model for protein embedding extraction.
        
        Args:
            model_name: ESM-2 model variant to use
        """
        try:
            import esm
            self.model, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
            self.model.eval()
            self.batch_converter = self.alphabet.get_batch_converter()
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = self.model.to(self.device)
            logger.info(f"ESM-2 model loaded successfully on {self.device}")
        except ImportError:
            logger.error("ESM library not found. Install with: pip install fair-esm")
            raise
    
    def extract_residue_embeddings(self, sequences: List[str], residue_positions: List[List[int]]) -> torch.Tensor:
        """
        Extract embeddings for specific residue positions from protein sequences.
        
        Args:
            sequences: List of protein sequences
            residue_positions: List of residue positions for each sequence
            
        Returns:
            Tensor of shape (n_sequences, n_residues, embedding_dim)
        """
        all_embeddings = []
        
        for seq, positions in zip(sequences, residue_positions):
            # Prepare sequence for ESM-2
            data = [("protein", seq)]
            batch_labels, batch_strs, batch_tokens = self.batch_converter(data)
            batch_tokens = batch_tokens.to(self.device)
            
            with torch.no_grad():
                results = self.model(batch_tokens, repr_layers=[33], return_contacts=False)
                token_representations = results["representations"][33]
                
                # Extract embeddings for specified residue positions
                residue_embeddings = []
                for pos in positions:
                    # ESM-2 uses 1-based indexing, add 1 for sequence start token
                    if pos < len(seq):
                        residue_embeddings.append(token_representations[0, pos + 1].cpu())
                
                # Concatenate residue embeddings
                if residue_embeddings:
                    all_embeddings.append(torch.cat(residue_embeddings, dim=0))
        
        return torch.stack(all_embeddings)

class LigandEmbeddingExtractor:
    """
    Extract ligand embeddings using ChemBERTa or molecular transformer models.
    
    ChemBERTa is a transformer model pre-trained on chemical SMILES strings
    for molecular representation learning.
    """
    
    def __init__(self, model_name: str = "seyonec/ChemBERTa-zinc-base-v1"):
        """
        Initialize ChemBERTa model for molecular embedding extraction.
        
        Args:
            model_name: ChemBERTa model to use
        """
        try:
            from transformers import AutoTokenizer, AutoModel
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.model.eval()
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = self.model.to(self.device)
            logger.info(f"ChemBERTa model loaded successfully on {self.device}")
        except ImportError:
            logger.error("Transformers library not found. Install with: pip install transformers")
            raise
    
    def extract_embeddings(self, smiles_list: List[str]) -> torch.Tensor:
        """
        Extract molecular embeddings from SMILES strings.
        
        Args:
            smiles_list: List of SMILES strings
            
        Returns:
            Tensor of molecular embeddings
        """
        embeddings = []
        
        for smiles in smiles_list:
            # Tokenize SMILES
            inputs = self.tokenizer(smiles, return_tensors="pt", truncation=True, 
                                  padding=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use CLS token embedding or mean pooling
                embedding = outputs.last_hidden_state.mean(dim=1).cpu()
                embeddings.append(embedding.squeeze())
        
        return torch.stack(embeddings)

class BindingAffinityDataset(Dataset):
    """
    Custom dataset for protein-ligand binding affinity prediction.
    
    Handles the combination of protein residue embeddings and ligand embeddings
    along with their corresponding binding affinity labels.
    """
    
    def __init__(self, protein_embeddings: torch.Tensor, ligand_embeddings: torch.Tensor, 
                 affinities: torch.Tensor, transform: Optional[callable] = None):
        """
        Initialize dataset.
        
        Args:
            protein_embeddings: Protein residue embeddings
            ligand_embeddings: Ligand molecular embeddings  
            affinities: Binding affinity labels
            transform: Optional data transforms
        """
        self.protein_embeddings = protein_embeddings
        self.ligand_embeddings = ligand_embeddings
        self.affinities = affinities
        self.transform = transform
        
        # Verify data consistency
        assert len(protein_embeddings) == len(ligand_embeddings) == len(affinities), \
            "All inputs must have the same number of samples"
    
    def __len__(self) -> int:
        return len(self.protein_embeddings)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (combined_features, affinity)
        """
        protein_emb = self.protein_embeddings[idx]
        ligand_emb = self.ligand_embeddings[idx]
        affinity = self.affinities[idx]
        
        # Combine protein and ligand embeddings
        combined_features = torch.cat([protein_emb.flatten(), ligand_emb.flatten()])
        
        if self.transform:
            combined_features = self.transform(combined_features)
        
        return combined_features, affinity

class MultiHeadAttentionTransformer(nn.Module):
    """
    Multi-head attention transformer for protein-ligand binding affinity prediction.
    
    This model uses transformer architecture with cross-attention mechanisms
    to learn complex interactions between protein and ligand representations.
    """
    
    def __init__(self, protein_dim: int, ligand_dim: int, d_model: int = 512, 
                 n_heads: int = 8, n_layers: int = 6, dropout: float = 0.1):
        """
        Initialize transformer model.
        
        Args:
            protein_dim: Dimension of protein embeddings
            ligand_dim: Dimension of ligand embeddings
            d_model: Model dimension
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
            dropout: Dropout rate for regularization
        """
        super(MultiHeadAttentionTransformer, self).__init__()
        
        self.protein_dim = protein_dim
        self.ligand_dim = ligand_dim
        self.d_model = d_model
        
        # Input projections
        self.protein_projection = nn.Linear(protein_dim, d_model)
        self.ligand_projection = nn.Linear(ligand_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model*4,
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Cross-attention for protein-ligand interaction
        self.cross_attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        
        # Regression head with advanced architecture
        self.regression_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights using Xavier/Glorot initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, protein_emb: torch.Tensor, ligand_emb: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            protein_emb: Protein embeddings
            ligand_emb: Ligand embeddings
            
        Returns:
            Predicted binding affinity
        """
        batch_size = protein_emb.size(0)
        
        # Project to model dimension
        protein_proj = self.protein_projection(protein_emb)
        ligand_proj = self.ligand_projection(ligand_emb)
        
        # Add sequence dimension if needed
        if len(protein_proj.shape) == 2:
            protein_proj = protein_proj.unsqueeze(1)
        if len(ligand_proj.shape) == 2:
            ligand_proj = ligand_proj.unsqueeze(1)
        
        # Apply positional encoding
        protein_proj = self.pos_encoder(protein_proj)
        ligand_proj = self.pos_encoder(ligand_proj)
        
        # Concatenate protein and ligand representations
        combined = torch.cat([protein_proj, ligand_proj], dim=1)
        
        # Pass through transformer encoder
        transformer_output = self.transformer_encoder(combined)
        
        # Apply cross-attention between protein and ligand
        query = transformer_output[:, :protein_proj.size(1)]  # Protein part
        key_value = transformer_output[:, protein_proj.size(1):]  # Ligand part
        
        attn_output, _ = self.cross_attention(query, key_value, key_value)
        
        # Global average pooling
        pooled = attn_output.mean(dim=1)
        
        # Regression prediction
        binding_affinity = self.regression_head(pooled)
        
        return binding_affinity.squeeze(-1)

class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer models.
    
    Adds sinusoidal positional information to input embeddings.
    """
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class BindingAffinityPredictor:
    """
    Main class for binding affinity prediction pipeline.
    
    Orchestrates the entire workflow from data preprocessing to model training,
    evaluation, and prediction generation.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the predictor with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize embedding extractors
        self.protein_extractor = ProteinEmbeddingExtractor()
        self.ligand_extractor = LigandEmbeddingExtractor()
        
        # Initialize model (will be set during training)
        self.model = None
        self.scaler = StandardScaler()
        
        # Training history
        self.training_history = {"train_loss": [], "val_loss": [], "val_metrics": []}
        
        logger.info(f"BindingAffinityPredictor initialized on {self.device}")
    
    def prepare_data(self, data: Dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepare and preprocess input data.
        
        Args:
            data: Dictionary containing sequences, ligands, residue positions, and affinities
            
        Returns:
            Tuple of (protein_embeddings, ligand_embeddings, affinities)
        """
        logger.info("Extracting protein embeddings using ESM-2...")
        protein_embeddings = self.protein_extractor.extract_residue_embeddings(
            data["sequences"], data["residue_positions"]
        )
        
        logger.info("Extracting ligand embeddings using ChemBERTa...")
        ligand_embeddings = self.ligand_extractor.extract_embeddings(data["ligands"])
        
        # Convert affinities to tensor
        affinities = torch.tensor(data["affinities"], dtype=torch.float32)
        
        logger.info(f"Data preparation complete. Shapes: "
                   f"Protein: {protein_embeddings.shape}, "
                   f"Ligand: {ligand_embeddings.shape}, "
                   f"Affinities: {affinities.shape}")
        
        return protein_embeddings, ligand_embeddings, affinities
    
    def create_model(self, protein_dim: int, ligand_dim: int) -> MultiHeadAttentionTransformer:
        """
        Create and initialize the transformer model.
        
        Args:
            protein_dim: Dimension of protein embeddings
            ligand_dim: Dimension of ligand embeddings
            
        Returns:
            Initialized model
        """
        model = MultiHeadAttentionTransformer(
            protein_dim=protein_dim,
            ligand_dim=ligand_dim,
            d_model=self.config.get("d_model", 512),
            n_heads=self.config.get("n_heads", 8),
            n_layers=self.config.get("n_layers", 6),
            dropout=self.config.get("dropout", 0.1)
        )
        
        model = model.to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logger.info(f"Model created with {total_params:,} total parameters "
                   f"({trainable_params:,} trainable)")
        
        return model
    
    def train(self, train_data: Dict, val_data: Optional[Dict] = None) -> None:
        """
        Train the binding affinity prediction model.
        
        Args:
            train_data: Training data dictionary
            val_data: Optional validation data dictionary
        """
        logger.info("Starting model training...")
        
        # Prepare training data
        protein_emb, ligand_emb, affinities = self.prepare_data(train_data)
        
        # Normalize affinities
        affinities_normalized = torch.tensor(
            self.scaler.fit_transform(affinities.numpy().reshape(-1, 1)).flatten(),
            dtype=torch.float32
        )
        
        # Create model
        protein_dim = protein_emb.size(1) * protein_emb.size(2) if len(protein_emb.shape) > 2 else protein_emb.size(1)
        ligand_dim = ligand_emb.size(1)
        self.model = self.create_model(protein_dim, ligand_dim)
        
        # Create dataset and dataloader
        dataset = BindingAffinityDataset(protein_emb, ligand_emb, affinities_normalized)
        dataloader = DataLoader(
            dataset, 
            batch_size=self.config.get("batch_size", 32),
            shuffle=True,
            num_workers=self.config.get("num_workers", 4)
        )
        
        # Prepare validation data if provided
        val_loader = None
        if val_data:
            val_protein_emb, val_ligand_emb, val_affinities = self.prepare_data(val_data)
            val_affinities_normalized = torch.tensor(
                self.scaler.transform(val_affinities.numpy().reshape(-1, 1)).flatten(),
                dtype=torch.float32
            )
            val_dataset = BindingAffinityDataset(val_protein_emb, val_ligand_emb, val_affinities_normalized)
            val_loader = DataLoader(val_dataset, batch_size=self.config.get("batch_size", 32))
        
        # Setup optimizer and scheduler
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.get("learning_rate", 1e-4),
            weight_decay=self.config.get("weight_decay", 1e-5)
        )
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        criterion = nn.MSELoss()
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        patience = self.config.get("patience", 10)
        
        for epoch in range(self.config.get("epochs", 100)):
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
            for features, targets in pbar:
                features, targets = features.to(self.device), targets.to(self.device)
                
                # Split features back into protein and ligand components
                protein_features = features[:, :protein_dim]
                ligand_features = features[:, protein_dim:]
                
                optimizer.zero_grad()
                predictions = self.model(protein_features, ligand_features)
                loss = criterion(predictions, targets)
                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                train_loss += loss.item()
                
                pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
            
            avg_train_loss = train_loss / len(dataloader)
            self.training_history["train_loss"].append(avg_train_loss)
            
            # Validation phase
            if val_loader:
                val_loss, val_metrics = self._evaluate(val_loader, protein_dim)
                self.training_history["val_loss"].append(val_loss)
                self.training_history["val_metrics"].append(val_metrics)
                
                scheduler.step(val_loss)
                
                logger.info(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, "
                           f"Val Loss: {val_loss:.4f}, Val R²: {val_metrics['r2']:.4f}")
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    self.save_model("best_model.pth")
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    logger.info(f"Early stopping triggered after {epoch+1} epochs")
                    break
            else:
                logger.info(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}")
        
        logger.info("Training completed!")
    
    def _evaluate(self, dataloader: DataLoader, protein_dim: int) -> Tuple[float, Dict]:
        """
        Evaluate model on validation data.
        
        Args:
            dataloader: Validation dataloader
            protein_dim: Dimension of protein features
            
        Returns:
            Tuple of (avg_loss, metrics_dict)
        """
        self.model.eval()
        total_loss = 0.0
        predictions = []
        targets = []
        
        with torch.no_grad():
            for features, batch_targets in dataloader:
                features, batch_targets = features.to(self.device), batch_targets.to(self.device)
                
                protein_features = features[:, :protein_dim]
                ligand_features = features[:, protein_dim:]
                
                batch_predictions = self.model(protein_features, ligand_features)
                loss = F.mse_loss(batch_predictions, batch_targets)
                
                total_loss += loss.item()
                predictions.extend(batch_predictions.cpu().numpy())
                targets.extend(batch_targets.cpu().numpy())
        
        avg_loss = total_loss / len(dataloader)
        
        # Calculate metrics
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        # Inverse transform for interpretable metrics
        predictions_original = self.scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
        targets_original = self.scaler.inverse_transform(targets.reshape(-1, 1)).flatten()
        
        metrics = {
            "mse": mean_squared_error(targets_original, predictions_original),
            "mae": mean_absolute_error(targets_original, predictions_original),
            "r2": r2_score(targets_original, predictions_original)
        }
        
        return avg_loss, metrics
    
    def predict(self, data: Dict) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            data: Data dictionary for prediction
            
        Returns:
            Array of predicted binding affinities
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        logger.info("Making predictions...")
        
        # Prepare data
        protein_emb, ligand_emb, _ = self.prepare_data(data)
        
        # Create dataset without targets
        affinities_dummy = torch.zeros(len(protein_emb))
        dataset = BindingAffinityDataset(protein_emb, ligand_emb, affinities_dummy)
        dataloader = DataLoader(dataset, batch_size=self.config.get("batch_size", 32))
        
        self.model.eval()
        predictions = []
        
        protein_dim = protein_emb.size(1) * protein_emb.size(2) if len(protein_emb.shape) > 2 else protein_emb.size(1)
        
        with torch.no_grad():
            for features, _ in dataloader:
                features = features.to(self.device)
                
                protein_features = features[:, :protein_dim]
                ligand_features = features[:, protein_dim:]
                
                batch_predictions = self.model(protein_features, ligand_features)
                predictions.extend(batch_predictions.cpu().numpy())
        
        # Inverse transform to original scale
        predictions = np.array(predictions)
        predictions_original = self.scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
        
        logger.info(f"Predictions completed. Shape: {predictions_original.shape}")
        
        return predictions_original
    
    def rank_mutants(self, predictions: np.ndarray, mutant_ids: List[str], 
                    output_file: str = "binding_affinity_rankings.csv") -> pd.DataFrame:
        """
        Rank mutants by predicted binding affinity and save results.
        
        Args:
            predictions: Predicted binding affinities
            mutant_ids: List of mutant identifiers
            output_file: Output CSV filename
            
        Returns:
            DataFrame with ranked results
        """
        # Create results dataframe
        results_df = pd.DataFrame({
            "Mutant_ID": mutant_ids,
            "Predicted_Binding_Affinity_kcal_mol": predictions,
            "Rank": range(1, len(predictions) + 1)
        })
        
        # Sort by binding affinity (lower is better for binding)
        results_df = results_df.sort_values("Predicted_Binding_Affinity_kcal_mol")
        results_df["Rank"] = range(1, len(results_df) + 1)
        
        # Add confidence categories
        results_df["Confidence_Category"] = pd.cut(
            results_df["Predicted_Binding_Affinity_kcal_mol"],
            bins=[-np.inf, -8, -6, -4, np.inf],
            labels=["Very Strong", "Strong", "Moderate", "Weak"]
        )
        
        # Save to CSV
        results_df.to_csv(output_file, index=False)
        logger.info(f"Results saved to {output_file}")
        
        # Print summary
        logger.info(f"Top 5 candidates for experimental validation:")
        for i, row in results_df.head().iterrows():
            logger.info(f"  {row['Rank']}. {row['Mutant_ID']}: "
                       f"{row['Predicted_Binding_Affinity_kcal_mol']:.2f} kcal/mol "
                       f"({row['Confidence_Category']})")
        
        return results_df
    
    def save_model(self, filepath: str) -> None:
        """
        Save model state and configuration.
        
        Args:
            filepath: Path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'scaler': self.scaler,
            'training_history': self.training_history
        }, filepath)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """
        Load model state and configuration.
        
        Args:
            filepath: Path to load the model from
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.config = checkpoint['config']
        self.scaler = checkpoint['scaler']
        self.training_history = checkpoint['training_history']
        
        # Recreate model architecture
        # Note: This assumes protein and ligand dimensions are stored in config
        self.model = self.create_model(
            checkpoint['config']['protein_dim'],
            checkpoint['config']['ligand_dim']
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        logger.info(f"Model loaded from {filepath}")
    
    def plot_training_history(self, save_path: str = "training_history.png") -> None:
        """
        Plot training history for analysis.
        
        Args:
            save_path: Path to save the plot
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        axes[0].plot(self.training_history["train_loss"], label="Training Loss")
        if self.training_history["val_loss"]:
            axes[0].plot(self.training_history["val_loss"], label="Validation Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Training and Validation Loss")
        axes[0].legend()
        axes[0].grid(True)
        
        # R² plot
        if self.training_history["val_metrics"]:
            r2_scores = [m["r2"] for m in self.training_history["val_metrics"]]
            axes[1].plot(r2_scores, label="Validation R²", color="green")
            axes[1].set_xlabel("Epoch")
            axes[1].set_ylabel("R² Score")
            axes[1].set_title("Validation R² Score")
            axes[1].legend()
            axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info(f"Training history plot saved to {save_path}")

def create_sample_data() -> Tuple[Dict, Dict, Dict]:
    """
    Create sample data for demonstration and testing.
    
    Returns:
        Tuple of (training_data, validation_data, prediction_data)
    """
    # Sample protein sequences (simplified for demonstration)
    sequences = [
        "MKVLWAALLVTFLAGCQAKVEQAVETEPEPELRQQTEWQSGQRWELALGR",  # Mutant 1
        "MKVLWAALLVTFLAGCQAKVEQAVETEPEPELRQQTEWQSGQRWELALGR",  # Mutant 2
        "MKVLWAALLVTFLAGCQAKVEQAVETEPEPELRQQTEWQSGQRWELALGR",  # Mutant 3
    ]
    
    # Active site residue positions (3 residues per mutant)
    residue_positions = [
        [48, 50, 108],  # Positions for mutant 1
        [48, 50, 108],  # Positions for mutant 2 
        [48, 50, 108],  # Positions for mutant 3
    ]
    
    # Sample SMILES for ligands A, B, C
    ligands_A = ["C1C(C(C(C(O1)CO)O)O)O"]  # Sugar A (actual substrate)
    ligands_B = ["C1C(C(C(C(O1)CO)O)O)N"]  # Sugar B (non-substrate)
    ligands_C = ["C1C(C(C(C(O1)CO)O)O)S"]  # Sugar C (prediction target)
    
    # Sample binding affinities (kcal/mol)
    affinities_A = [-6.5, -7.2, -5.8]
    affinities_B = [-4.2, -4.8, -3.9]
    
    # Training data (mutants with ligands A and B)
    train_data = {
        "sequences": sequences + sequences,
        "residue_positions": residue_positions + residue_positions,
        "ligands": ligands_A + ligands_B,
        "affinities": affinities_A + affinities_B
    }
    
    # Validation data (subset of training data)
    val_data = {
        "sequences": sequences[:2],
        "residue_positions": residue_positions[:2],
        "ligands": ligands_A[:2],
        "affinities": affinities_A[:2]
    }
    
    # Prediction data (all mutants with ligand C)
    pred_data = {
        "sequences": sequences,
        "residue_positions": residue_positions,
        "ligands": ligands_C * len(sequences)
    }
    
    return train_data, val_data, pred_data

def main():
    """
    Main execution function demonstrating the complete pipeline.
    """
    # Set random seeds for reproducibility
    set_random_seeds(42)
    
    # Configuration
    config = {
        "d_model": 512,
        "n_heads": 8,
        "n_layers": 6,
        "dropout": 0.1,
        "batch_size": 16,
        "learning_rate": 1e-4,
        "weight_decay": 1e-5,
        "epochs": 50,
        "patience": 10,
        "num_workers": 2,
    }
    
    logger.info("Starting Binding Affinity Prediction Pipeline")
    logger.info("=" * 60)
    
    # Create sample data (replace with your actual data loading)
    train_data, val_data, pred_data = create_sample_data()
    
    # Initialize predictor
    predictor = BindingAffinityPredictor(config)
    
    # Train model
    predictor.train(train_data, val_data)
    
    # Make predictions for ligand C
    mutant_ids = ["Mutant_1_A48X_D50Y_D108Z", "Mutant_2_A48X_D50Y_D108Z", "Mutant_3_A48X_D50Y_D108Z"]
    predictions = predictor.predict(pred_data)
    
    # Rank mutants and save results
    results_df = predictor.rank_mutants(predictions, mutant_ids)
    
    # Plot training history
    predictor.plot_training_history()
    
    # Save final model
    predictor.save_model("final_binding_affinity_model.pth")
    
    logger.info("Pipeline completed successfully!")
    logger.info(f"Best candidates for experimental validation:")
    logger.info(results_df.head()[["Mutant_ID", "Predicted_Binding_Affinity_kcal_mol", "Confidence_Category"]])

if __name__ == "__main__":
    main()