
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from pi_head import PiHead

class Backbone(nn.Module):
    """
    Encoder condiviso che estrae feature a 1/8 della risoluzione.
    Output: [B, 128, H/8, W/8]
    """
    def __init__(self):
        super().__init__()
        # Carica un ResNet-18 pre-addestrato
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # Estraiamo le feature fino a layer2 (che downscala di 8x)
        self.features = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, # output: [B, 64, H/4, W/4]
            resnet.layer2  # output: [B, 128, H/8, W/8]
        )
        self.out_channels = 128 # Importante per il PiHead!
        
    def forward(self, x):
        return self.features(x)
    
class DensityDecoder(nn.Module):
    """
    Placeholder per il tuo "VLM-EBC".
    Prende le feature (già mascherate) e predice i conteggi per blocco.
    Input: [B, 128, H/8, W/8]
    Output: [B, 1, H/8, W/8]
    """
    def __init__(self, in_channels=128):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.ReLU() # I conteggi non possono essere negativi
        )
        
    def forward(self, masked_features):
        # Predice direttamente i block_counts
        return self.decoder(masked_features)
    

# model.py (continua)

class CrowdCounter(nn.Module):
    """
    Modello End-to-End che combina Backbone, Pi-Head e Density Decoder.
    """
    def __init__(self, lambda_pi=0.5):
        super().__init__()
        self.lambda_pi = lambda_pi
        
        # 1. Encoder Condiviso
        self.backbone = Backbone()
        
        # 2. Pi-Head (Filtro)
        # Inizializziamo il tuo PiHead con i canali corretti
        self.pi_head = PiHead(
            in_channels=self.backbone.out_channels, # 128
            hidden_channels=128,
            num_layers=3
        )
        
        # 3. Density Decoder (Contatore)
        self.density_decoder = DensityDecoder(
            in_channels=self.backbone.out_channels # 128
        )
        
        # Definiamo la loss per il contatore (VLM-EBC)
        self.density_loss_fn = nn.MSELoss() # Ottimo per i conteggi

    def forward(self, x):
        """
        Esegue la pipeline completa.
        """
        # 1. Estrattore condiviso
        features = self.backbone(x) # [B, 128, H/8, W/8]
        
        # 2. Ramo 1: Pi-Head (Filtro)
        # pi_output è un dict {'pi': ..., 'mask': ...}
        pi_output = self.pi_head(features, return_mask=True)
        
        # 3. Applica la maschera (Zero Tokenization)
        # Usiamo la funzione del tuo script!
        masked_features = self.pi_head.get_masked_features(features, pi_output)
        
        # 4. Ramo 2: Density Decoder (Contatore)
        # Predice i conteggi solo dalle feature rilevanti
        pred_block_counts = self.density_decoder(masked_features)
        
        # Ritorna tutto il necessario per la loss
        return {
            "pred_block_counts": pred_block_counts, # Per la loss di densità
            "pi_pred": pi_output['pi']              # Per la loss del pi-head
        }

    def compute_total_loss(self, predictions, batch):
        """
        Calcola la loss combinata (multi-task).
        """
        # --- Target dal tuo Dataset ---
        gt_counts = batch["gt_block_counts"]
        gt_pi = batch["gt_pi"]
        
        # --- Predizioni dal Modello ---
        pred_counts = predictions["pred_block_counts"]
        pi_pred = predictions["pi_pred"]

        # Aggiungi dimensione "canale" ai target per la loss
        if gt_counts.dim() == 3:
            gt_counts = gt_counts.unsqueeze(1) # [B, H/8, W/8] -> [B, 1, H/8, W/8]
        
        # --- Loss 1: Density/Count Loss (del VLM-EBC) ---
        loss_density = self.density_loss_fn(pred_counts, gt_counts)
        
        # --- Loss 2: Pi-Head Loss (BCE) ---
        # Usa la funzione di loss dal tuo script!
        loss_pi = self.pi_head.compute_loss(
            pi_pred, 
            gt_pi,
            lambda_pi=1.0 # Applichiamo il lambda fuori
        )
        
        # --- Loss Totale ---
        total_loss = loss_density + self.lambda_pi * loss_pi
        
        return {
            "total_loss": total_loss,
            "loss_density": loss_density,
            "loss_pi": loss_pi
        }