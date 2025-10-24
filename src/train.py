# train.py (pseudo-codice)

from model import CrowdCounter
from dataset import CrowdDataset #
from torch.utils.data import DataLoader
import torch.optim as optim

# 1. Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Usa il tuo Dataset ---
train_dataset = CrowdDataset(
    root="path/to/shanghaitech_a", 
    dataset_name="shanghaitech_a",
    split="train",
    block_size=8, # Deve corrispondere al downsampling (H/8)
    img_size=512
)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

# --- Usa il tuo Modello Combinato ---
model = CrowdCounter(lambda_pi=0.5).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# 2. Training Loop
for epoch in range(num_epochs):
    for batch in train_loader:
        # Sposta i dati su GPU
        images = batch["image"].to(device)
        batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

        # 1. Forward pass
        predictions = model(images)
        
        # 2. Calcolo Loss
        losses = model.compute_total_loss(predictions, batch)
        
        # 3. Backward pass
        optimizer.zero_grad()
        losses["total_loss"].backward()
        optimizer.step()
        
        print(f"Epoch {epoch}, Loss Tot: {losses['total_loss'].item():.4f}, "
              f"Loss Density: {losses['loss_density'].item():.4f}, "
              f"Loss Pi: {losses['loss_pi'].item():.4f}")