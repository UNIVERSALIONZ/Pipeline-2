"""
Simplified Demo Script for Binding Affinity Prediction
======================================================

This script demonstrates the core functionality without requiring
heavy dependencies like ESM-2 or ChemBERTa. It uses mock embeddings
to show the complete workflow.

Usage: python demo_simple.py
"""

import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

def set_random_seeds(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

class SimplifiedTransformerModel(nn.Module):
    """
    Simplified transformer model for binding affinity prediction.
    Uses mock embeddings instead of actual ESM-2/ChemBERTa.
    """
    
    def __init__(self, protein_dim=384, ligand_dim=128, d_model=256, n_heads=8):
        super(SimplifiedTransformerModel, self).__init__()
        
        # Input projections
        self.protein_proj = nn.Linear(protein_dim, d_model)
        self.ligand_proj = nn.Linear(ligand_dim, d_model)
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        
        # Regression head
        self.regression_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, d_model // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 4, 1)
        )
    
    def forward(self, protein_emb, ligand_emb):
        # Project embeddings
        protein_proj = self.protein_proj(protein_emb).unsqueeze(1)
        ligand_proj = self.ligand_proj(ligand_emb).unsqueeze(1)
        
        # Combine protein and ligand
        combined = torch.cat([protein_proj, ligand_proj], dim=1)
        
        # Self-attention
        attn_output, _ = self.attention(combined, combined, combined)
        
        # Global average pooling
        pooled = attn_output.mean(dim=1)
        
        # Regression
        return self.regression_head(pooled).squeeze(-1)

class MockEmbeddingExtractor:
    """Mock embedding extractor for demonstration."""
    
    def __init__(self, protein_dim=384, ligand_dim=128):
        self.protein_dim = protein_dim
        self.ligand_dim = ligand_dim
    
    def extract_protein_embeddings(self, sequences, residue_positions):
        """Extract mock protein embeddings."""
        embeddings = []
        for seq, positions in zip(sequences, residue_positions):
            # Create mock embeddings based on sequence characteristics
            seq_features = []
            for pos in positions:
                if pos < len(seq):
                    # Simple encoding based on amino acid properties
                    aa = seq[pos]
                    aa_props = self._get_aa_properties(aa)
                    seq_features.extend(aa_props)
                else:
                    seq_features.extend([0.0] * (self.protein_dim // 3))
            
            # Pad or truncate to correct dimension
            while len(seq_features) < self.protein_dim:
                seq_features.append(0.0)
            seq_features = seq_features[:self.protein_dim]
            
            embeddings.append(seq_features)
        
        return torch.tensor(embeddings, dtype=torch.float32)
    
    def extract_ligand_embeddings(self, smiles_list):
        """Extract mock ligand embeddings."""
        embeddings = []
        for smiles in smiles_list:
            # Simple molecular descriptors based on SMILES
            features = self._get_molecular_features(smiles)
            embeddings.append(features)
        
        return torch.tensor(embeddings, dtype=torch.float32)
    
    def _get_aa_properties(self, aa):
        """Get simple amino acid properties."""
        # Basic physicochemical properties (hydrophobicity, charge, size)
        aa_props = {
            'A': [1.8, 0, 1], 'R': [-4.5, 1, 4], 'N': [-3.5, 0, 2], 'D': [-3.5, -1, 2],
            'C': [2.5, 0, 2], 'Q': [-3.5, 0, 3], 'E': [-3.5, -1, 3], 'G': [-0.4, 0, 1],
            'H': [-3.2, 0.5, 3], 'I': [4.5, 0, 3], 'L': [3.8, 0, 3], 'K': [-3.9, 1, 4],
            'M': [1.9, 0, 3], 'F': [2.8, 0, 4], 'P': [-1.6, 0, 2], 'S': [-0.8, 0, 1],
            'T': [-0.7, 0, 2], 'W': [-0.9, 0, 5], 'Y': [-1.3, 0, 4], 'V': [4.2, 0, 2]
        }
        props = aa_props.get(aa, [0, 0, 1])
        # Expand to fill dimension
        expanded = props * (self.protein_dim // 9)
        while len(expanded) < self.protein_dim // 3:
            expanded.append(0.0)
        return expanded[:self.protein_dim // 3]
    
    def _get_molecular_features(self, smiles):
        """Get simple molecular features from SMILES."""
        features = []
        
        # Basic molecular descriptors
        features.append(len(smiles))  # Molecular size
        features.append(smiles.count('C'))  # Carbon count
        features.append(smiles.count('O'))  # Oxygen count
        features.append(smiles.count('N'))  # Nitrogen count
        features.append(smiles.count('S'))  # Sulfur count
        features.append(smiles.count('('))  # Branching
        features.append(smiles.count('='))  # Double bonds
        features.append(smiles.count('#'))  # Triple bonds
        
        # Pad to correct dimension
        while len(features) < self.ligand_dim:
            features.append(0.0)
        
        return features[:self.ligand_dim]

class SimpleBindingAffinityDataset(Dataset):
    """Simple dataset for protein-ligand binding affinity."""
    
    def __init__(self, protein_embeddings, ligand_embeddings, affinities):
        self.protein_embeddings = protein_embeddings
        self.ligand_embeddings = ligand_embeddings
        self.affinities = affinities
    
    def __len__(self):
        return len(self.protein_embeddings)
    
    def __getitem__(self, idx):
        return (self.protein_embeddings[idx], 
                self.ligand_embeddings[idx], 
                self.affinities[idx])

def create_demo_data():
    """Create demonstration data."""
    
    # Sample HAS mutant sequences (shortened for demo)
    sequences = [
        "MKVLWAALLVTFLAGCQAKVEQAVETEPEPELRQQTEWQSGQRWELALGRPHQVPVQLQRVAAHRDGRVSVAQLRFNASLHRRWPRTSFGPPEKRFDDDFYELLFSRPLHRLADFSSKHLFQRLAELKAELQAEVFIRFGDRPPETAEAIEKVLQAVVQAVKKAGGPPGPPPPPAALATPDE",
        "MKVLWAALLVTFLAGCQAKVEQAVETEPEPELRQQTEWQSGQRWELALGRPHQVPVQLQRVAAHRDGRVSVAQLRFNASLHRRWPRTSFGPPEKRFDDDFYELLFSRPLHRLADFSSKHLFQRLAELKAELQAEVFIRFGDRPPETAEAIEKVLQAVVQAVKKAGGPPGPPPPPAALATPDE",
        "MKVLWAALLVTFLAGCQAKVEQAVETEPEPELRQQTEWQSGQRWELALGRPHQVPVQLQRVAAHRDGRVSVAQLRFNASLHRRWPRTSFGPPEKRFDDDFYELLFSRPLHRLADFSSKHLFQRLAELKAELQAEVFIRFGDRPPETAEAIEKVLQAVVQAVKKAGGPPGPPPPPAALATPDE",
        "MKVLWAALLVTFLAGCQAKVEQAVETEPEPELRQQTEWQSGQRWELALGRPHQVPVQLQRVAAHRDGRVSVAQLRFNASLHRRWPRTSFGPPEKRFDDDFYELLFSRPLHRLADFSSKHLFQRLAELKAELQAEVFIRFGDRPPETAEAIEKVLQAVVQAVKKAGGPPGPPPPPAALATPDE",
        "MKVLWAALLVTFLAGCQAKVEQAVETEPEPELRQQTEWQSGQRWELALGRPHQVPVQLQRVAAHRDGRVSVAQLRFNASLHRRWPRTSFGPPEKRFDDDFYELLFSRPLHRLADFSSKHLFQRLAELKAELQAEVFIRFGDRPPETAEAIEKVLQAVVQAVKKAGGPPGPPPPPAALATPDE"
    ]
    
    # Active site positions (0-indexed)
    residue_positions = [
        [47, 49, 107],  # Original HAS positions 48, 50, 108
        [47, 49, 107],
        [47, 49, 107], 
        [47, 49, 107],
        [47, 49, 107]
    ]
    
    # Ligand SMILES
    ligands = {
        "A": "C1C(C(C(C(O1)CO)O)O)O",      # Sugar A (glucose-like, actual substrate)
        "B": "C1C(C(C(C(O1)CO)O)O)N",      # Sugar B (amino sugar, non-substrate)
        "C": "C1C(C(C(C(O1)CO)O)O)S"       # Sugar C (thio sugar, prediction target)
    }
    
    # Mutant identifiers
    mutant_ids = [
        "HAS_A48V_D50E_D108A",
        "HAS_A48L_D50K_D108F", 
        "HAS_A48I_D50R_D108Y",
        "HAS_A48M_D50Q_D108W",
        "HAS_A48T_D50N_D108H"
    ]
    
    # Mock binding affinities (kcal/mol) for training
    # Real data would come from docking or experimental measurements
    binding_affinities = {
        # Substrate A (stronger binding expected)
        ("HAS_A48V_D50E_D108A", "A"): -7.2,
        ("HAS_A48L_D50K_D108F", "A"): -6.8,
        ("HAS_A48I_D50R_D108Y", "A"): -6.5,
        ("HAS_A48M_D50Q_D108W", "A"): -7.0,
        ("HAS_A48T_D50N_D108H", "A"): -6.3,
        
        # Non-substrate B (weaker binding expected)
        ("HAS_A48V_D50E_D108A", "B"): -4.8,
        ("HAS_A48L_D50K_D108F", "B"): -4.2,
        ("HAS_A48I_D50R_D108Y", "B"): -3.9,
        ("HAS_A48M_D50Q_D108W", "B"): -4.5,
        ("HAS_A48T_D50N_D108H", "B"): -3.7,
    }
    
    return {
        "sequences": sequences,
        "positions": residue_positions,
        "ligands": ligands,
        "affinities": binding_affinities,
        "mutant_ids": mutant_ids
    }

def prepare_training_data(data):
    """Prepare training data from demo data."""
    sequences = []
    positions = []
    ligands = []
    affinities = []
    
    # Combine data from ligands A and B
    for i, mutant_id in enumerate(data["mutant_ids"]):
        # Add ligand A data
        if (mutant_id, "A") in data["affinities"]:
            sequences.append(data["sequences"][i])
            positions.append(data["positions"][i])
            ligands.append(data["ligands"]["A"])
            affinities.append(data["affinities"][(mutant_id, "A")])
        
        # Add ligand B data
        if (mutant_id, "B") in data["affinities"]:
            sequences.append(data["sequences"][i])
            positions.append(data["positions"][i])
            ligands.append(data["ligands"]["B"])
            affinities.append(data["affinities"][(mutant_id, "B")])
    
    return sequences, positions, ligands, affinities

def prepare_prediction_data(data):
    """Prepare prediction data for ligand C."""
    sequences = data["sequences"]
    positions = data["positions"]
    ligands = [data["ligands"]["C"]] * len(sequences)
    
    return sequences, positions, ligands

def train_model(model, train_loader, val_loader, num_epochs=20):
    """Train the binding affinity model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)
    
    train_losses = []
    val_losses = []
    val_r2_scores = []
    
    print(f"Training on {device}")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for protein_emb, ligand_emb, affinity in train_loader:
            protein_emb = protein_emb.to(device)
            ligand_emb = ligand_emb.to(device)
            affinity = affinity.to(device)
            
            optimizer.zero_grad()
            prediction = model(protein_emb, ligand_emb)
            loss = criterion(prediction, affinity)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        predictions = []
        targets = []
        
        with torch.no_grad():
            for protein_emb, ligand_emb, affinity in val_loader:
                protein_emb = protein_emb.to(device)
                ligand_emb = ligand_emb.to(device)
                affinity = affinity.to(device)
                
                prediction = model(protein_emb, ligand_emb)
                loss = criterion(prediction, affinity)
                val_loss += loss.item()
                
                predictions.extend(prediction.cpu().numpy())
                targets.extend(affinity.cpu().numpy())
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # Calculate R²
        r2 = r2_score(targets, predictions)
        val_r2_scores.append(r2)
        
        scheduler.step(avg_val_loss)
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch:2d}: Train Loss: {avg_train_loss:.4f}, "
                  f"Val Loss: {avg_val_loss:.4f}, Val R²: {r2:.4f}")
    
    return train_losses, val_losses, val_r2_scores

def make_predictions(model, pred_loader):
    """Make predictions with the trained model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    predictions = []
    
    with torch.no_grad():
        for protein_emb, ligand_emb, _ in pred_loader:
            protein_emb = protein_emb.to(device)
            ligand_emb = ligand_emb.to(device)
            
            prediction = model(protein_emb, ligand_emb)
            predictions.extend(prediction.cpu().numpy())
    
    return np.array(predictions)

def plot_results(train_losses, val_losses, val_r2_scores, predictions, mutant_ids):
    """Plot training results and predictions."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Training loss
    axes[0, 0].plot(train_losses, label='Training Loss')
    axes[0, 0].plot(val_losses, label='Validation Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # R² score
    axes[0, 1].plot(val_r2_scores, label='Validation R²', color='green')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('R² Score')
    axes[0, 1].set_title('Validation R² Score')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Binding affinity predictions
    indices = np.arange(len(predictions))
    bars = axes[1, 0].bar(indices, predictions, alpha=0.7)
    axes[1, 0].set_xlabel('Mutant')
    axes[1, 0].set_ylabel('Predicted Binding Affinity (kcal/mol)')
    axes[1, 0].set_title('Predicted Binding Affinities for Ligand C')
    axes[1, 0].set_xticks(indices)
    axes[1, 0].set_xticklabels([f"M{i+1}" for i in indices], rotation=45)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Color bars by binding strength
    for i, bar in enumerate(bars):
        if predictions[i] < -7:
            bar.set_color('darkgreen')
        elif predictions[i] < -5:
            bar.set_color('green')
        elif predictions[i] < -3:
            bar.set_color('orange')
        else:
            bar.set_color('red')
    
    # Ranking table
    ranking_data = sorted(zip(mutant_ids, predictions), key=lambda x: x[1])
    axes[1, 1].axis('tight')
    axes[1, 1].axis('off')
    
    table_data = []
    for i, (mutant, affinity) in enumerate(ranking_data):
        confidence = "Very Strong" if affinity < -7 else "Strong" if affinity < -5 else "Moderate" if affinity < -3 else "Weak"
        table_data.append([i+1, mutant.replace("HAS_", ""), f"{affinity:.2f}", confidence])
    
    table = axes[1, 1].table(cellText=table_data,
                            colLabels=['Rank', 'Mutant', 'ΔG (kcal/mol)', 'Confidence'],
                            cellLoc='center',
                            loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    axes[1, 1].set_title('Binding Affinity Rankings')
    
    plt.tight_layout()
    plt.savefig('demo_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main demo function."""
    print("🧬 HAS Mutant Binding Affinity Prediction Demo")
    print("=" * 50)
    
    # Set random seed for reproducibility
    set_random_seeds(42)
    
    # Create demo data
    print("📊 Creating demonstration data...")
    demo_data = create_demo_data()
    
    # Initialize mock embedding extractor
    extractor = MockEmbeddingExtractor()
    
    # Prepare training data
    print("🔧 Preparing training data...")
    train_sequences, train_positions, train_ligands, train_affinities = prepare_training_data(demo_data)
    
    # Extract embeddings
    print("🧠 Extracting embeddings...")
    train_protein_emb = extractor.extract_protein_embeddings(train_sequences, train_positions)
    train_ligand_emb = extractor.extract_ligand_embeddings(train_ligands)
    train_affinities_tensor = torch.tensor(train_affinities, dtype=torch.float32)
    
    # Normalize affinities
    scaler = StandardScaler()
    train_affinities_norm = scaler.fit_transform(np.array(train_affinities).reshape(-1, 1)).flatten()
    train_affinities_norm = torch.tensor(train_affinities_norm, dtype=torch.float32)
    
    # Create datasets
    train_dataset = SimpleBindingAffinityDataset(train_protein_emb[:8], train_ligand_emb[:8], train_affinities_norm[:8])
    val_dataset = SimpleBindingAffinityDataset(train_protein_emb[8:], train_ligand_emb[8:], train_affinities_norm[8:])
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4)
    
    # Create and train model
    print("🤖 Creating and training model...")
    model = SimplifiedTransformerModel()
    train_losses, val_losses, val_r2_scores = train_model(model, train_loader, val_loader, num_epochs=30)
    
    # Prepare prediction data
    print("🔮 Preparing prediction data for ligand C...")
    pred_sequences, pred_positions, pred_ligands = prepare_prediction_data(demo_data)
    pred_protein_emb = extractor.extract_protein_embeddings(pred_sequences, pred_positions)
    pred_ligand_emb = extractor.extract_ligand_embeddings(pred_ligands)
    pred_affinities_dummy = torch.zeros(len(pred_protein_emb))
    
    pred_dataset = SimpleBindingAffinityDataset(pred_protein_emb, pred_ligand_emb, pred_affinities_dummy)
    pred_loader = DataLoader(pred_dataset, batch_size=4)
    
    # Make predictions
    print("🎯 Making predictions...")
    predictions_norm = make_predictions(model, pred_loader)
    predictions = scaler.inverse_transform(predictions_norm.reshape(-1, 1)).flatten()
    
    # Create results dataframe
    results_df = pd.DataFrame({
        'Mutant_ID': demo_data['mutant_ids'],
        'Predicted_Binding_Affinity_kcal_mol': predictions,
        'Rank': range(1, len(predictions) + 1)
    })
    
    # Sort by binding affinity
    results_df = results_df.sort_values('Predicted_Binding_Affinity_kcal_mol')
    results_df['Rank'] = range(1, len(results_df) + 1)
    
    # Add confidence categories
    results_df['Confidence_Category'] = pd.cut(
        results_df['Predicted_Binding_Affinity_kcal_mol'],
        bins=[-np.inf, -7, -5, -3, np.inf],
        labels=['Very Strong', 'Strong', 'Moderate', 'Weak']
    )
    
    # Save results
    results_df.to_csv('demo_binding_affinity_rankings.csv', index=False)
    
    # Plot results
    print("📈 Generating plots...")
    plot_results(train_losses, val_losses, val_r2_scores, predictions, demo_data['mutant_ids'])
    
    # Print summary
    print("\n🎉 Demo Complete!")
    print("-" * 40)
    print("📊 Top 3 Candidates for Experimental Validation:")
    
    for idx, row in results_df.head(3).iterrows():
        print(f"  {row['Rank']}. {row['Mutant_ID']}")
        print(f"     Predicted ΔG: {row['Predicted_Binding_Affinity_kcal_mol']:.2f} kcal/mol")
        print(f"     Confidence: {row['Confidence_Category']}")
        print()
    
    print(f"📁 Results saved to: demo_binding_affinity_rankings.csv")
    print(f"📈 Plots saved to: demo_results.png")
    
    print("\n🔬 Ready for Experimental Validation!")
    print("Consider the top-ranked mutants for isothermal titration calorimetry (ITC)")

if __name__ == "__main__":
    main()