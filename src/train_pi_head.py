# train.py
import os
import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Importa i tuoi moduli
try:
    from .pi_head import PiHead, AdaptivePiHead
    from .dataset import CrowdDataset
    from .test_pi_head import tiny_backbone, get_device # Riusiamo le utility
except ImportError:
    from pi_head import PiHead, AdaptivePiHead
    from dataset import CrowdDataset
    from test_pi_head import tiny_backbone, get_device


def run_training(args):
    device = get_device()
    print(f"Usando il device: {device}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Setup Dataloader (usando il tuo dataset.py)
    # Nota: block_size=8 è hardcoded nel backbone (H/8, W/8)
    train_dataset = CrowdDataset(
        root=args.data_root,
        dataset_name=args.dataset_name,
        split="train",
        block_size=8 
    )
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers,
        pin_memory=True
    )
    print(f"Trovate {len(train_dataset)} immagini in {args.data_root}/train")

    # 2. Setup Modelli
    backbone = tiny_backbone(out_ch=args.feature_channels).to(device)
    
    HeadCls = AdaptivePiHead if args.adaptive else PiHead
    pi_head = HeadCls(
        in_channels=args.feature_channels,
        hidden_channels=args.hidden_channels,
        num_layers=args.num_layers,
    ).to(device)
    
    # 3. Setup Ottimizzatore
    params = list(backbone.parameters()) + list(pi_head.parameters())
    optimizer = optim.Adam(params, lr=args.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda')) # Per mixed precision

    # 4. Training Loop
    backbone.train()
    pi_head.train()
    
    print("Inizio addestramento...")
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        start_time = time.time()
        
        for i, batch in enumerate(train_loader):
            images = batch["image"].to(device)
            # Questo è il target [B, Hb, Wb] (0=Occupato, 1=Vuoto)
            pi_target = batch["gt_pi"].to(device) 

            optimizer.zero_grad(set_to_none=True)
            
            # Forward pass (con mixed precision se su Cuda)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == 'cuda')):
                features = backbone(images)      # [B, C, H/8, W/8]
                out = pi_head(features, return_mask=False) # Vogliamo 'pi'
                pi_pred = out['pi']              # [B, 1, H/8, W/8]
                
                # Calcola loss (usando la funzione aggiornata in pi_head.py)
                loss = pi_head.compute_loss(pi_pred, pi_target, lambda_pi=args.lambda_pi)

            # Backward pass
            if device.type == 'cuda':
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else: # Per MPS (M1) o CPU
                loss.backward()
                optimizer.step()
            
            epoch_loss += loss.item()
            
            if (i + 1) % 20 == 0:
                print(f"  [Epoch {epoch+1}/{args.epochs}, Step {i+1}/{len(train_loader)}] Loss: {loss.item():.4f}")

        end_time = time.time()
        avg_loss = epoch_loss / len(train_loader)
        print(f"--- Epoch {epoch+1} completata in {end_time-start_time:.2f}s ---")
        print(f"Loss media: {avg_loss:.4f}")
        if torch.isnan(torch.tensor(avg_loss)):
            print("Loss è NaN. Interrompo l'addestramento.")
            break
        
    # 5. Salva i modelli
    print("Addestramento completato.")
    model_name = f"{args.dataset_name}_{'adaptive' if args.adaptive else 'fixed'}.pth"
    torch.save(backbone.state_dict(), output_dir / f"backbone_{model_name}")
    torch.save(pi_head.state_dict(), output_dir / f"pi_head_{model_name}")
    print(f"Modelli salvati in {output_dir}")


def build_parser():
    p = argparse.ArgumentParser("Training script per PiHead (con dataset.py)")
    
    # Path
    p.add_argument("--data-root", type=str, required=True, 
                   help="Path alla cartella root del dataset (es. ../data/ShanghaiTech)")
    p.add_argument("--dataset-name", type=str, default="shanghai_a", 
                   help="Nome dataset (es. shanghai_a, shanghai_b)")
    p.add_argument("--output-dir", type=str, default="checkpoints", 
                   help="Cartella dove salvare i modelli addestrati")

    # Parametri di training
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    p.add_argument("--lambda-pi", type=float, default=1.0, help="Peso della PiHead loss")
    p.add_argument("--num-workers", type=int, default=2)

    # Parametri del modello
    p.add_argument("--feature-channels", type=int, default=256)
    p.add_argument("--hidden-channels", type=int, default=256)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--adaptive", action="store_true", help="Usa AdaptivePiHead")

    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    # Esempio: --data-root ../data/SHA --dataset-name shanghai_a
    # Assumendo che esista: ../data/SHA/train/images e ../data/SHA/train/labels
    if "shanghai_a" in args.dataset_name.lower():
        args.data_root = os.path.join(args.data_root, "part_A")
    elif "shanghai_b" in args.dataset_name.lower():
        args.data_root = os.path.join(args.data_root, "part_B")
        
    run_training(args)