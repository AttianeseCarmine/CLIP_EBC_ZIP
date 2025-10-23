import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
import cv2


class PiHeadVisualizer:
    """
    Classe per visualizzare i risultati del π-Head
    """
    
    def __init__(self, figsize=(15, 5)):
        self.figsize = figsize
    
    def visualize_prediction(self, image, pi_map, mask, density_gt=None, save_path=None, title=None):
        """
        Visualizza immagine originale, π map, maschera e opzionalmente la density GT
        
        Args:
            image (torch.Tensor or np.ndarray): Immagine [H, W, 3] o [3, H, W]
            pi_map (torch.Tensor): π predictions [1, H, W] o [H, W]
            mask (torch.Tensor): Binary mask [1, H, W] o [H, W]
            density_gt (torch.Tensor, optional): Ground truth density map
            save_path (str, optional): Path per salvare la figura
        """
        # Converti tutto in numpy
        image = self._to_numpy(image)
        pi_map = self._to_numpy(pi_map)
        mask = self._to_numpy(mask)
        
        # Normalizza immagine se necessario
        if image.max() > 1:
            image = image / 255.0
        
        # Determina numero di subplot
        n_plots = 4 if density_gt is not None else 3
        
        fig, axes = plt.subplots(1, n_plots, figsize=(5*n_plots, 5))
        
        # 1. Immagine originale
        axes[0].imshow(image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # 2. π map (probabilità di essere vuoto)
        im1 = axes[1].imshow(pi_map, cmap='hot', vmin=0, vmax=1)
        axes[1].set_title(f'π Map\n(Empty Probability)')
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1], fraction=0.046)
        
        # 3. Maschera binaria
        im2 = axes[2].imshow(mask, cmap='gray', vmin=0, vmax=1)
        axes[2].set_title('Binary Mask\n(0=Empty, 1=Occupied)')
        axes[2].axis('off')
        plt.colorbar(im2, ax=axes[2], fraction=0.046)
        
        # 4. Density GT (se disponibile)
        if density_gt is not None:
            density_gt = self._to_numpy(density_gt)
            im3 = axes[3].imshow(density_gt, cmap='jet')
            axes[3].set_title('Ground Truth Density')
            axes[3].axis('off')
            plt.colorbar(im3, ax=axes[3], fraction=0.046)

        if title:
            fig.suptitle(title, fontsize=16, y=1.02)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figura salvata in: {save_path}")
        
        plt.show()
    
    def visualize_masked_regions(self, image, mask, alpha=0.5, save_path=None):
        """
        Sovrappone la maschera all'immagine per evidenziare regioni vuote/occupate
        
        Args:
            image (torch.Tensor or np.ndarray): Immagine originale
            mask (torch.Tensor): Binary mask (1=occupied, 0=empty)
            alpha (float): Trasparenza dell'overlay
            save_path (str, optional): Path per salvare
        """
        image = self._to_numpy(image)
        mask = self._to_numpy(mask)
        
        if image.max() > 1:
            image = image / 255.0
        
        # Crea overlay colorato
        # Rosso per regioni vuote, verde per regioni occupate
        overlay = np.zeros_like(image)
        overlay[:, :, 0] = (1 - mask)  # Rosso per empty (mask=0)
        overlay[:, :, 1] = mask         # Verde per occupied (mask=1)
        
        # Blend
        blended = image * (1 - alpha) + overlay * alpha
        
        plt.figure(figsize=(10, 10))
        plt.imshow(blended)
        plt.title('Masked Regions\n(Red=Empty, Green=Occupied)')
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
    
    def visualize_block_statistics(self, pi_map, mask, block_size=8):
        """
        Visualizza statistiche per blocco dell'immagine
        
        Args:
            pi_map (torch.Tensor): π predictions
            mask (torch.Tensor): Binary mask
            block_size (int): Dimensione dei blocchi da analizzare
        """
        pi_map = self._to_numpy(pi_map)
        mask = self._to_numpy(mask)
        
        H, W = pi_map.shape
        
        # Dividi in blocchi
        num_blocks_h = H // block_size
        num_blocks_w = W // block_size
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Mappa dei blocchi vuoti
        block_empty = np.zeros((num_blocks_h, num_blocks_w))
        for i in range(num_blocks_h):
            for j in range(num_blocks_w):
                block = mask[i*block_size:(i+1)*block_size, 
                           j*block_size:(j+1)*block_size]
                block_empty[i, j] = (block.mean() < 0.5)  # Vuoto se >50% è 0
        
        im1 = axes[0].imshow(block_empty, cmap='RdYlGn', vmin=0, vmax=1)
        axes[0].set_title(f'Block Occupancy Map\n(Block size: {block_size}x{block_size})')
        axes[0].set_xlabel('Block X')
        axes[0].set_ylabel('Block Y')
        plt.colorbar(im1, ax=axes[0])
        
        # Istogramma delle probabilità π
        axes[1].hist(pi_map.flatten(), bins=50, edgecolor='black', alpha=0.7)
        axes[1].axvline(0.5, color='red', linestyle='--', label='Threshold (0.5)')
        axes[1].set_xlabel('π (Empty Probability)')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Distribution of π values')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Stampa statistiche
        empty_percentage = (mask == 0).sum() / mask.size * 100
        print(f"\n=== Block Statistics ===")
        print(f"Total blocks: {num_blocks_h * num_blocks_w}")
        print(f"Empty blocks: {block_empty.sum():.0f}")
        print(f"Occupied blocks: {(1-block_empty).sum():.0f}")
        print(f"Empty percentage: {empty_percentage:.2f}%")
        print(f"Mean π: {pi_map.mean():.3f}")
        print(f"Std π: {pi_map.std():.3f}")
    
    def compare_thresholds(self, pi_map, thresholds=[0.3, 0.5, 0.7], save_path=None):
        """
        Confronta l'effetto di diverse soglie sul π-Head
        
        Args:
            pi_map (torch.Tensor): π predictions
            thresholds (list): Lista di soglie da testare
            save_path (str): Path per salvare
        """
        pi_map = self._to_numpy(pi_map)
        
        n_thresholds = len(thresholds)
        fig, axes = plt.subplots(1, n_thresholds + 1, figsize=(5*(n_thresholds+1), 5))
        
        # π map originale
        im0 = axes[0].imshow(pi_map, cmap='hot', vmin=0, vmax=1)
        axes[0].set_title('Original π Map')
        axes[0].axis('off')
        plt.colorbar(im0, ax=axes[0], fraction=0.046)
        
        # Maschere con diverse soglie
        for idx, thresh in enumerate(thresholds):
            mask = (pi_map < thresh).astype(float)
            empty_pct = (mask == 0).sum() / mask.size * 100
            
            im = axes[idx+1].imshow(mask, cmap='gray', vmin=0, vmax=1)
            axes[idx+1].set_title(f'Threshold = {thresh}\nEmpty: {empty_pct:.1f}%')
            axes[idx+1].axis('off')
            plt.colorbar(im, ax=axes[idx+1], fraction=0.046)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
    
    def _to_numpy(self, tensor):
        """Converte tensor in numpy array"""
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.detach().cpu().numpy()
        
        # Rimuovi batch dimension se presente
        if tensor.ndim == 4:
            tensor = tensor[0]
        
        # Rimuovi channel dimension se singolo canale
        if tensor.ndim == 3 and tensor.shape[0] == 1:
            tensor = tensor[0]
        
        # Se è un'immagine [C, H, W] -> [H, W, C]
        if tensor.ndim == 3 and tensor.shape[0] in [1, 3]:
            tensor = np.transpose(tensor, (1, 2, 0))
            if tensor.shape[2] == 1:
                tensor = tensor[:, :, 0]
        
        return tensor


# Esempio di utilizzo completo
def demo_pi_head_visualization():
    """Demo completo del π-Head con visualizzazione"""
    from pi_head import PiHead
    
    print("=== Demo π-Head Visualization ===\n")
    
    # 1. Crea il modulo
    pi_head = PiHead(in_channels=256, hidden_channels=128)
    pi_head.eval()
    
    # 2. Simula feature maps
    batch_size = 1
    features = torch.randn(batch_size, 256, 64, 64)
    
    # 3. Forward pass
    with torch.no_grad():
        output = pi_head(features, return_mask=True)
    
    # 4. Simula un'immagine
    image = torch.rand(3, 64, 64)
    
    # 5. Visualizza
    visualizer = PiHeadVisualizer()
    
    print("1. Visualizing predictions...")
    visualizer.visualize_prediction(
        image=image,
        pi_map=output['pi'][0],
        mask=output['mask'][0]
    )
    
    print("\n2. Visualizing masked regions...")
    visualizer.visualize_masked_regions(
        image=image,
        mask=output['mask'][0]
    )
    
    print("\n3. Block statistics...")
    visualizer.visualize_block_statistics(
        pi_map=output['pi'][0, 0],
        mask=output['mask'][0, 0],
        block_size=8
    )
    
    print("\n4. Comparing thresholds...")
    visualizer.compare_thresholds(
        pi_map=output['pi'][0, 0],
        thresholds=[0.3, 0.5, 0.7]
    )


if __name__ == "__main__":
    demo_pi_head_visualization()