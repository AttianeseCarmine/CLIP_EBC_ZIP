import os
import glob
import math
from typing import Optional, Dict, Any, Tuple
import warnings

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import cv2


# ============ Utility ============
def resize_density_map_keep_count(d: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """
    Ridimensiona una density map preservando il conteggio totale (integrale).
    Usa INTER_NEAREST per mantenere la sparsità (zero rimane zero).
    
    Args:
        d: Density map 2D
        out_h: Altezza output
        out_w: Larghezza output
    
    Returns:
        Density map ridimensionata con conteggio preservato
    """
    assert d.ndim == 2, "density map deve essere 2D"
    h0, w0 = d.shape
    if (h0, w0) == (out_h, out_w):
        return d.astype(np.float32)

    d = d.astype(np.float32)
    d_res = cv2.resize(d, (out_w, out_h), interpolation=cv2.INTER_NEAREST)

    # Correzione di scala per conservare l'integrale
    scale = (h0 / out_h) * (w0 / out_w)
    d_res *= scale

    # Pulizia numerica: azzera quasi-zeri
    eps = np.finfo(np.float32).eps
    d_res[abs(d_res) < eps] = 0.0
    return d_res


def block_sum(arr: np.ndarray, bh: int, bw: int) -> np.ndarray:
    """
    Somma per blocchi bh×bw (padding a destra/basso se non divisibile).
    
    Args:
        arr: Array 2D
        bh: Altezza blocco
        bw: Larghezza blocco
    
    Returns:
        Array con somme per blocco [n_blocks_h, n_blocks_w]
    """
    assert arr.ndim == 2
    H, W = arr.shape
    Hp = math.ceil(H / bh) * bh
    Wp = math.ceil(W / bw) * bw
    if (Hp, Wp) != (H, W):
        arr = np.pad(arr, ((0, Hp - H), (0, Wp - W)), mode="constant")
        H, W = Hp, Wp
    nh, nw = H // bh, W // bw
    return arr.reshape(nh, bh, nw, bw).sum(axis=(1, 3))


def safe_load_image(path: str) -> np.ndarray:
    """Carica immagine RGB in numpy [H,W,3], dtype=uint8."""
    with Image.open(path) as im:
        return np.array(im.convert("RGB"))


def get_transforms(
    split: str = "train",
    img_size: int = 512,
    val_size: int = 1024,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
):
    """
    Ritorna una funzione trasformazione compatibile con Albumentations.
    
    Args:
        split: 'train', 'val', o 'test'
        img_size: Dimensione crop per training
        val_size: Dimensione massima per validation
        mean: Media per normalizzazione
        std: Deviazione standard per normalizzazione
    
    Returns:
        Trasformazione Albumentations o fallback
    """
    try:
        import albumentations as A
        from albumentations.pytorch import ToTensorV2

        if split == "train":
            tfm = A.Compose([
                A.SmallestMaxSize(max_size=img_size, interpolation=cv2.INTER_AREA), 
                A.RandomCrop(height=img_size, width=img_size, always_apply=True), 
                A.HorizontalFlip(p=0.5),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ])
        else:  # val/test
            tfm = A.Compose([
                A.LongestMaxSize(max_size=val_size, interpolation=cv2.INTER_AREA),
                A.CenterCrop(height=img_size, width=img_size, always_apply=True),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ])
        return tfm

    except ImportError:
        warnings.warn("Albumentations non disponibile. Uso trasformazione minimale.")
        # Fallback: identità (senza normalizzazione)
        class _FallbackTransform:
            def __init__(self, img_size, mean, std):
                self.img_size = img_size
                self.mean = np.array(mean).reshape(3, 1, 1)
                self.std = np.array(std).reshape(3, 1, 1)
            
            def __call__(self, *, image, mask=None):
                # Center crop
                h, w = image.shape[:2]
                if h > self.img_size or w > self.img_size:
                    start_h = max(0, (h - self.img_size) // 2)
                    start_w = max(0, (w - self.img_size) // 2)
                    image = image[start_h:start_h+self.img_size, start_w:start_w+self.img_size]
                    if mask is not None:
                        mask = mask[start_h:start_h+self.img_size, start_w:start_w+self.img_size]
                
                # Normalizza
                img = image.astype(np.float32) / 255.0
                img = (img - self.mean.reshape(1, 1, 3)) / self.std.reshape(1, 1, 3)
                img_t = torch.from_numpy(img).permute(2, 0, 1)  # [3,H,W]
                
                out = {"image": img_t}
                if mask is not None:
                    out["mask"] = torch.from_numpy(mask.astype(np.float32))
                return out
        
        return _FallbackTransform(img_size, mean, std)


# ============ Dataset ============

class CrowdDataset(Dataset):
    """
    Loader per crowd counting con label .npy (density map).
    
    Struttura attesa:
        root/
          train/
            images/*.jpg|png
            labels/*.npy
          val/
            images/*.jpg|png
            labels/*.npy
    
    Args:
        root: Percorso root del dataset
        dataset_name: Nome dataset (es. 'shanghaitech_a')
        split: 'train', 'val', o 'test'
        block_size: Dimensione blocco in pixel per block_counts
        img_size: Dimensione crop training
        val_size: Dimensione massima validation
        pi_threshold: Soglia conteggio/pixel per classificare blocco come vuoto
        transform: Trasformazione custom (se None usa default)
        validate_pairs: Se True, valida che ogni immagine abbia il suo .npy
    """
    def __init__(
        self,
        root: str,
        dataset_name: str,
        split: str = "train",
        block_size: int = 8,
        img_size: int = 512,
        val_size: int = 1024,
        pi_threshold: Optional[float] = None,
        transform: Optional[Any] = None,
        validate_pairs: bool = True,
    ):
        super().__init__()
        self.root = root
        self.dataset_name = dataset_name.lower()
        self.split = split
        self.block_size = int(block_size)
        self.pi_threshold = pi_threshold  # None = usa 0 esatto
        
        # Setup directories
        if "shanghai" in self.dataset_name:
            self.img_dir = os.path.join(root, split, "images")
            self.ann_dir = os.path.join(root, split, "labels")
        else:
            raise ValueError(f"Dataset '{dataset_name}' non supportato. "
                           "Estendi il codice per UCF-QNRF/NWPU.")

        if not os.path.exists(self.img_dir):
            raise FileNotFoundError(f"Cartella immagini non trovata: {self.img_dir}")
        if not os.path.exists(self.ann_dir):
            raise FileNotFoundError(f"Cartella labels non trovata: {self.ann_dir}")

        # Trova immagini
        self.img_paths = sorted(
            glob.glob(os.path.join(self.img_dir, "*.jpg")) +
            glob.glob(os.path.join(self.img_dir, "*.jpeg")) +
            glob.glob(os.path.join(self.img_dir, "*.png"))
        )
        if len(self.img_paths) == 0:
            raise RuntimeError(f"Nessuna immagine trovata in {self.img_dir}")

        # Valida coppie immagine-label
        if validate_pairs:
            self._validate_pairs()

        # Setup transforms
        self.transform = transform or get_transforms(
            split=split, 
            img_size=img_size, 
            val_size=val_size
        )

    def _validate_pairs(self):
        """Valida che ogni immagine abbia il suo .npy corrispondente."""
        missing = []
        for img_path in self.img_paths:
            try:
                self._match_npy_for_image(img_path)
            except FileNotFoundError:
                missing.append(os.path.basename(img_path))
        
        if missing:
            warnings.warn(
                f"Trovate {len(missing)} immagini senza .npy corrispondente: "
                f"{missing[:5]}{'...' if len(missing) > 5 else ''}"
            )

    def _match_npy_for_image(self, img_path: str) -> str:
        """
        Trova il .npy corrispondente all'immagine.
        Prova: <base>.npy, rimuove 'IMG_', prova zfill 3/4/5, glob.
        
        Args:
            img_path: Percorso immagine
        
        Returns:
            Percorso file .npy
        
        Raises:
            FileNotFoundError: Se .npy non trovato
        """
        base = os.path.splitext(os.path.basename(img_path))[0]
        
        # 1) Match diretto
        p = os.path.join(self.ann_dir, f"{base}.npy")
        if os.path.exists(p):
            return p
        
        # 2) Senza prefisso IMG_
        alt = base.replace("IMG_", "")
        for cand in (alt, alt.zfill(3), alt.zfill(4), alt.zfill(5)):
            p = os.path.join(self.ann_dir, f"{cand}.npy")
            if os.path.exists(p):
                return p
        
        # 3) Glob rilassato
        matches = sorted(glob.glob(os.path.join(self.ann_dir, f"*{alt}*.npy")))
        if matches:
            return matches[0]
        
        raise FileNotFoundError(
            f".npy per '{base}' non trovato in {self.ann_dir}"
        )

    def __len__(self) -> int:
        return len(self.img_paths)

    def _compute_pi_target(self, block_counts: np.ndarray) -> np.ndarray:
        """
        Computa il target π (probabilità di essere vuoto) per ogni blocco.
        
        Strategia:
        1. Se pi_threshold è None: usa 0 esatto (blocco vuoto se count == 0)
        2. Se pi_threshold è fornito: usa soglia su conteggio/pixel
        
        Args:
            block_counts: Conteggi per blocco [Hb, Wb]
        
        Returns:
            π target [Hb, Wb]: 1.0 = vuoto, 0.0 = occupato
        """
        bs = self.block_size
        
        if self.pi_threshold is None:
            # Criterio rigoroso: vuoto solo se esattamente 0
            pi_gt = (block_counts == 0.0).astype(np.float32)
        else:
            # Criterio con soglia: normalizza per pixel
            per_pix = block_counts / float(bs * bs)
            pi_gt = (per_pix <= self.pi_threshold).astype(np.float32)
        
        return pi_gt

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_path = self.img_paths[idx]
        base_name = os.path.splitext(os.path.basename(img_path))[0]

        # 1) Carica immagine
        img = safe_load_image(img_path)  # [H,W,3] uint8

        # 2) Carica density map
        npy_path = self._match_npy_for_image(img_path)
        density = np.load(npy_path)
        if density.ndim == 3:
            density = density[..., 0]
        density = density.astype(np.float32)

        # 3) Allinea density a img (richiesto da Albumentations)
        if density.shape != img.shape[:2]:
            density = resize_density_map_keep_count(
                density, img.shape[0], img.shape[1]
            )
        
        # 4) Applica transforms (stesse geo-trasformazioni)
        if self.transform is not None:
            augmented = self.transform(image=img, mask=density)
            img_t = augmented["image"]
            density_t = augmented["mask"]
            
            # Converti a numpy se necessario
            if isinstance(density_t, torch.Tensor):
                density_np = density_t.cpu().numpy().astype(np.float32)
            else:
                density_np = density_t.astype(np.float32)
        else:
            # Fallback manuale
            img_t = torch.from_numpy(
                (img.astype(np.float32) / 255.0)
            ).permute(2, 0, 1)
            density_np = density

        # 5) Safety: riallinea density a dimensioni esatte
        H, W = img_t.shape[-2], img_t.shape[-1]
        if density_np.shape != (H, W):
            density_np = resize_density_map_keep_count(density_np, H, W)

        # 6) Computa target blockwise
        bs = self.block_size
        block_counts = block_sum(density_np, bs, bs).astype(np.float32)
        pi_gt = self._compute_pi_target(block_counts)

        sample = {
            "image": img_t,
            "gt_density": torch.from_numpy(density_np),
            "gt_block_counts": torch.from_numpy(block_counts),
            "gt_pi": torch.from_numpy(pi_gt),
            "count": float(density_np.sum()),  # Conteggio totale
            "base_name": base_name,
            "image_path": img_path,
            "label_path": npy_path,
        }
        return sample


# ============ Utility per analisi dataset ============

def analyze_dataset_statistics(dataset: CrowdDataset, num_samples: int = 100):
    """
    Analizza statistiche del dataset per calibrare iperparametri.
    
    Args:
        dataset: Dataset da analizzare
        num_samples: Numero di campioni da analizzare
    
    Returns:
        Dict con statistiche
    """
    empty_ratios = []
    counts = []
    block_densities = []
    
    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        block_counts = sample["gt_block_counts"].numpy()
        pi = sample["gt_pi"].numpy()
        
        empty_ratios.append(pi.mean())
        counts.append(sample["count"])
        block_densities.extend(block_counts.flatten())
    
    stats = {
        "mean_empty_ratio": np.mean(empty_ratios),
        "std_empty_ratio": np.std(empty_ratios),
        "mean_count": np.mean(counts),
        "median_count": np.median(counts),
        "block_density_percentiles": {
            "p25": np.percentile(block_densities, 25),
            "p50": np.percentile(block_densities, 50),
            "p75": np.percentile(block_densities, 75),
            "p90": np.percentile(block_densities, 90),
        }
    }
    
    print("=== Dataset Statistics ===")
    print(f"Empty ratio: {stats['mean_empty_ratio']:.3f} ± {stats['std_empty_ratio']:.3f}")
    print(f"Avg count: {stats['mean_count']:.1f} (median: {stats['median_count']:.1f})")
    print(f"Block density percentiles: {stats['block_density_percentiles']}")
    
    return stats