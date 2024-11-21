import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModel, AutoTokenizer
import numpy as np
import pandas as pd

# Load ProBERT model for protein embeddings
probert_model_name = "Rostlab/prot_bert"
tokenizer = AutoTokenizer.from_pretrained(probert_model_name)
probert_model = AutoModel.from_pretrained(probert_model_name)

# Define a function to generate embeddings for protein sequences
def generate_protein_embeddings(sequence):
    inputs = tokenizer(sequence, return_tensors="pt")
    with torch.no_grad():
        outputs = probert_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)  # Mean pooling

# Define a function to generate SMILES embeddings using a simple character-level encoding
def generate_smiles_embeddings(smiles):
    smiles_vocab = list(set(''.join(smiles)))
    smiles_vocab_size = len(smiles_vocab)
    max_len = max(len(s) for s in smiles)
    one_hot_vectors = np.zeros((len(smiles), max_len, smiles_vocab_size))
    
    char_to_index = {char: idx for idx, char in enumerate(smiles_vocab)}
    for i, smile in enumerate(smiles):
        for j, char in enumerate(smile):
            one_hot_vectors[i, j, char_to_index[char]] = 1
    
    return torch.tensor(one_hot_vectors, dtype=torch.float32)

# Load protein and ligand data
protein_sequences = ["AAGKKL..."]  # Replace with your protein sequences
ligands_smiles = ["CC(C)C..."]  # Replace with your ligand SMILES

# Generate protein and ligand embeddings
protein_embeddings = torch.cat([generate_protein_embeddings(seq) for seq in protein_sequences], dim=0)
ligand_embeddings = generate_smiles_embeddings(ligands_smiles)

# Cross Attention Model for Docking Prediction
class CrossAttentionModel(nn.Module):
    def __init__(self, protein_dim, ligand_dim, output_dim):
        super(CrossAttentionModel, self).__init__()
        self.protein_proj = nn.Linear(protein_dim, output_dim)
        self.ligand_proj = nn.Linear(ligand_dim, output_dim)
        self.attn_layer = nn.MultiheadAttention(embed_dim=output_dim, num_heads=8)
        self.fc_layers = nn.Sequential(
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, 1)
        )
    
    def forward(self, protein_embedding, ligand_embedding):
        protein_embed_proj = self.protein_proj(protein_embedding)
        ligand_embed_proj = self.ligand_proj(ligand_embedding)
        
        # Using cross-attention
        attn_output, _ = self.attn_layer(protein_embed_proj.unsqueeze(0), 
                                         ligand_embed_proj.unsqueeze(0), 
                                         ligand_embed_proj.unsqueeze(0))
        return self.fc_layers(attn_output.squeeze(0))

# Model Hyperparameters
protein_dim = protein_embeddings.shape[1]
ligand_dim = ligand_embeddings.shape[2]
output_dim = 128

# Instantiate the model
model = CrossAttentionModel(protein_dim, ligand_dim, output_dim)

# Training setup
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 5

# Labels (binding affinities)
labels = torch.tensor([0.5, 1.2], dtype=torch.float32)  # Replace with actual binding affinity labels

# Training Loop
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    
    predictions = []
    for i in range(len(protein_sequences)):
        prediction = model(protein_embeddings[i], ligand_embeddings[i])
        predictions.append(prediction)
    
    predictions = torch.cat(predictions, dim=0)
    loss = criterion(predictions, labels)
    
    loss.backward()
    optimizer.step()
    
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

# Evaluation
model.eval()
with torch.no_grad():
    predictions = []
    for i in range(len(protein_sequences)):
        prediction = model(protein_embeddings[i], ligand_embeddings[i])
        predictions.append(prediction)
    predictions = torch.cat(predictions, dim=0)
    print("Predictions:", predictions)
