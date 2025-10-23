import torch
import torch.nn as nn
import torch.nn.functional as F


class PiHead(nn.Module):
    """
    π-Head Module per identificare regioni vuote dell'immagine.
    
    Questo modulo predice la probabilità π che un blocco/patch dell'immagine
    sia "vuoto" (senza persone), permettendo di mascherare regioni irrilevanti
    e concentrare il modello solo dove ci sono persone.
    
    Args:
        in_channels (int): Numero di canali in input dalle feature maps
        hidden_channels (int): Numero di canali nascosti
        num_layers (int): Numero di layer convoluzionali
        dropout (float): Dropout rate
        threshold (float): Soglia per considerare un blocco "vuoto" (default: 0.5)
    """
    
    def __init__(
        self, 
        in_channels=256,
        hidden_channels=128,
        num_layers=3,
        dropout=0.1,
        threshold=0.5
    ):
        super(PiHead, self).__init__()
        
        self.threshold = threshold
        
        # Convolutional layers per processare le feature
        layers = []
        current_channels = in_channels
        
        for i in range(num_layers - 1):
            layers.extend([
                nn.Conv2d(current_channels, hidden_channels, 
                         kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(hidden_channels),
                nn.ReLU(inplace=True),
                nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
            ])
            current_channels = hidden_channels
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # Final layer per predire π (probabilità di essere vuoto)
        # Output: 1 canale con valori in [0, 1]
        self.pi_predictor = nn.Sequential(
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
            nn.Sigmoid()  # Output in [0, 1]
        )
        
        # Optional: Refinement head per migliorare le predizioni
        self.refine = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Inizializzazione dei pesi"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, features, return_mask=True, refine=True):
        """
        Forward pass del π-Head
        
        Args:
            features (torch.Tensor): Feature maps [B, C, H, W]
            return_mask (bool): Se True, ritorna anche la maschera binaria
            refine (bool): Se True, applica il refinement
        
        Returns:
            dict: Contiene:
                - 'pi': Probabilità di essere vuoto [B, 1, H, W] (valori in [0,1])
                - 'mask': Maschera binaria [B, 1, H, W] (0=vuoto, 1=occupato)
                - 'empty_ratio': Percentuale di regioni vuote
        """
        # Estrai feature
        x = self.feature_extractor(features)
        
        # Predici π
        pi = self.pi_predictor(x)
        
        # Optional refinement
        if refine:
            pi = self.refine(pi)
        
        output = {'pi': pi}
        
        # Crea maschera binaria
        if return_mask:
            # mask = 1 dove NON è vuoto (π < threshold)
            # mask = 0 dove è vuoto (π >= threshold)
            mask = (pi < self.threshold).float()
            output['mask'] = mask
            
            # Calcola la percentuale di regioni vuote
            empty_ratio = (pi >= self.threshold).float().mean()
            output['empty_ratio'] = empty_ratio
        
        return output
    
    def get_masked_features(self, features, pi_output):
        """
        Applica la maschera alle feature per focalizzare solo su regioni occupate
        
        Args:
            features (torch.Tensor): Feature maps originali [B, C, H, W]
            pi_output (dict): Output del forward pass del π-Head
        
        Returns:
            torch.Tensor: Feature mascherate [B, C, H, W]
        """
        mask = pi_output['mask']
        
        # Espandi la maschera per matchare i canali delle feature
        # mask: [B, 1, H, W] -> [B, C, H, W]
        mask_expanded = mask.expand_as(features)
        
        # Applica la maschera (moltiplica per mantenere solo regioni occupate)
        masked_features = features * mask_expanded
        
        return masked_features
    
    def compute_loss(self, pi_pred, pi_target, lambda_pi=0.5):
        """
        Calcola la loss per il π-Head
        
        Args:
            pi_pred (torch.Tensor): Predizioni π [B, 1, H, W]
            pi_target (torch.Tensor): Ground truth [B, 1, H, W] (0=vuoto, 1=occupato)
            lambda_pi (float): Peso della loss
        
        Returns:
            torch.Tensor: Loss value
        """
        # Il target (pi_target) è già calcolato perfettamente dal Dataset
        # gt_pi è (0=vuoto, 1=occupato)
        # pi_pred è (probabilità di essere vuoto)
        
        # Per farli coincidere: invertiamo il pi_target
        # Vogliamo che il modello preda pi=1.0 quando il target è 0 (vuoto)
        # Vogliamo che il modello preda pi=0.0 quando il target è 1 (occupato)
        
        # Il tuo dataset.py crea: pi_gt = (block_counts == 0.0)
        # Quindi pi_gt = 1.0 (vuoto), 0.0 (occupato)
        # La predizione pi_pred è la probabilità di essere VUOTO
        # Sono già allineati! pi_target è la probabilità di essere vuoto.
        
        # CORREZIONE: Controlliamo la logica del tuo dataset
        # pi_gt = (block_counts == 0.0).astype(np.float32)
        # OK, quindi:
        # pi_gt = 1.0  <-- Blocco VUOTO
        # pi_gt = 0.0  <-- Blocco OCCUPATO
        
        # E il tuo pi_head.py predice:
        # # Final layer per predire π (probabilità di essere vuoto)
        # ...
        # nn.Sigmoid()
        # OK, quindi:
        # pi_pred è la probabilità di essere VUOTO.
        
        # I target e le predizioni sono perfettamente allineati.
        
        # Dobbiamo solo aggiungere la dimensione del canale a pi_target
        if pi_target.dim() == 3:
            pi_target = pi_target.unsqueeze(1) # [B, H, W] -> [B, 1, H, W]
        
        # Binary Cross Entropy Loss
        bce_loss = F.binary_cross_entropy(pi_pred, pi_target, reduction='mean')
        
        # Optional: Regularizzazione
        mean_pi = pi_pred.mean()
        balance_loss = torch.abs(mean_pi - 0.5)
        
        total_loss = lambda_pi * (bce_loss + 0.1 * balance_loss)
        
        return total_loss


class AdaptivePiHead(PiHead):
    """
    Versione avanzata del π-Head con soglia adattiva
    
    Invece di usare una soglia fissa, apprende la soglia ottimale
    durante il training.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Parametro apprendibile per la soglia
        self.threshold_param = nn.Parameter(torch.tensor(0.0))  # logit space
    
    def get_threshold(self):
        """Ottiene la soglia adattiva in [0, 1]"""
        return torch.sigmoid(self.threshold_param)
    
    def forward(self, features, return_mask=True, refine=True):
        """Forward con soglia adattiva"""
        # Estrai feature
        x = self.feature_extractor(features)
        pi = self.pi_predictor(x)
        
        if refine:
            pi = self.refine(pi)
        
        output = {'pi': pi}
        
        if return_mask:
            # Usa la soglia adattiva
            adaptive_threshold = self.get_threshold()
            mask = (pi < adaptive_threshold).float()
            output['mask'] = mask
            output['threshold'] = adaptive_threshold
            output['empty_ratio'] = (pi >= adaptive_threshold).float().mean()
        
        return output


# Esempio di utilizzo
if __name__ == "__main__":
    # Crea il modulo
    pi_head = PiHead(
        in_channels=256,
        hidden_channels=128,
        num_layers=3,
        threshold=0.5
    )
    
    # Input fittizio: feature maps da un encoder
    batch_size = 2
    features = torch.randn(batch_size, 256, 64, 64)
    
    # Forward pass
    output = pi_head(features, return_mask=True)
    
    print(f"π shape: {output['pi'].shape}")  # [2, 1, 64, 64]
    print(f"Mask shape: {output['mask'].shape}")  # [2, 1, 64, 64]
    print(f"Empty ratio: {output['empty_ratio']:.3f}")
    
    # Applica maschera alle feature
    masked_features = PiHead.get_masked_features(features, output)
    print(f"Masked features shape: {masked_features.shape}")  # [2, 256, 64, 64]
    
    # Calcola loss (esempio con density map fittizia)
    density_gt = torch.rand(batch_size, 1, 64, 64)
    loss = PiHead.compute_loss(output['pi'], density_gt)
    print(f"Loss: {loss.item():.4f}")