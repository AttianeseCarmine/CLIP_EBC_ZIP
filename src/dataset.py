
import os
import glob
import math
from typing import Optional, Dict, Any

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
    """
    assert d.ndim == 2, "density map deve essere 2D"
    h0, w0 = d.shape
    if (h0, w0) == (out_h, out_w):
        return d.astype(np.float32)

    d = d.astype(np.float32)
    # nearest per non “spalmare” la massa sui vicini (sparsità)
    d_res = cv2.resize(d, (out_w, out_h), interpolation=cv2.INTER_NEAREST)

    # correzione di scala per conservare l'integrale
    scale = (h0 / out_h) * (w0 / out_w)
    d_res *= scale

    # pulizia numerica minima: azzera veri zeri “quasi-zeri” con eps di macchina
    # (non è una costante scelta a mano; dipende solo dal dtype)
    eps = np.finfo(np.float32).eps
    d_res[abs(d_res) < eps] = 0.0
    return d_res

def otsu_threshold(x: np.ndarray) -> float:
    """
    Soglia di Otsu su array 1D non-negativo.
    Restituisce 0 se la varianza totale è nulla (caso degenerato).
    """
    x = x.astype(np.float32).ravel()
    x = x[x >= 0]
    if x.size == 0:
        return 0.0
    # istogramma automatico (256 bin su range dei dati)
    hist, bin_edges = np.histogram(x, bins=256, range=(x.min(), x.max() if x.max()>x.min() else x.min()+1.0))
    p = hist.astype(np.float64)
    p /= p.sum() if p.sum() > 0 else 1.0
    omega = np.cumsum(p)
    mu = np.cumsum(p * (bin_edges[:-1] + bin_edges[1:]) * 0.5)
    mu_t = mu[-1] if omega[-1] > 0 else 0.0
    sigma_b2 = (mu_t * omega - mu)**2 / (omega * (1.0 - omega) + 1e-12)
    k = int(np.nanargmax(sigma_b2))
    thr = 0.5 * (bin_edges[k] + bin_edges[k+1])
    return float(thr)

def block_sum(arr: np.ndarray, bh: int, bw: int) -> np.ndarray:
    """Somma per blocchi bh×bw (padding a destra/basso se non divisibile)."""
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


def lazy_get_transforms(split: str):
    """
    Ritorna una funzione trasformazione compatibile con Albumentations
    (dict con keys 'image' e 'mask'). Se Albumentations non è installato,
    restituisce una trasformazione identità minimale.
    """
    try:
        import albumentations as A
        from albumentations.pytorch import ToTensorV2

        if split == "train":
            tfm = A.Compose([
                A.LongestMaxSize(max_size=1024, interpolation=cv2.INTER_AREA),
                A.RandomCrop(height=512, width=512, always_apply=True),
                A.HorizontalFlip(p=0.5),
                A.Normalize(mean=(0.485, 0.456, 0.406),
                            std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])
        else:  # val/test
            tfm = A.Compose([
                A.LongestMaxSize(max_size=1024, interpolation=cv2.INTER_AREA),
                A.CenterCrop(height=512, width=512, always_apply=True),
                A.Normalize(mean=(0.485, 0.456, 0.406),
                            std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])
        return tfm

    except Exception:
        # Fallback: identità (senza normalizzazione), ritorna tensori torch
        def _fallback_transform(image, mask=None):
            img = image.astype(np.float32) / 255.0
            img_t = torch.from_numpy(img).permute(2, 0, 1)  # [3,H,W]
            out = {"image": img_t}
            if mask is not None:
                out["mask"] = torch.from_numpy(mask.astype(np.float32))
            return out

        class _Wrapper:
            def __call__(self, *, image, mask=None):
                return _fallback_transform(image, mask)

        return _Wrapper()


# ============ Dataset ============

class CrowdDataset(Dataset):
    """
    Loader per crowd counting (stile ZIP / CLIP-EBC) con label .npy (density map).
    Struttura attesa:
        root/
          train/
            images/*.jpg|png
            labels/*.npy
          val/
            images/*.jpg|png
            labels/*.npy
    """
    def __init__(
        self,
        root: str,
        dataset_name: str,
        split: str = "train",
        block_size: int = 8,
        transform: Optional[Any] = None,
    ):
        super().__init__()
        self.root = root
        self.dataset_name = dataset_name.lower()
        self.split = split
        self.block_size = int(block_size)
        self.transform = transform or lazy_get_transforms(split)

        if "shanghai" in self.dataset_name:
            # <root>/<split>/images, <root>/<split>/labels
            self.img_dir = os.path.join(root, split, "images")
            self.ann_dir = os.path.join(root, split, "labels")
        else:
            # puoi estendere qui per UCF-QNRF / NWPU
            raise ValueError(f"Dataset '{dataset_name}' non ancora supportato in questo loader.")

        if not os.path.exists(self.img_dir):
            raise FileNotFoundError(f"Cartella immagini non trovata: {self.img_dir}")
        if not os.path.exists(self.ann_dir):
            raise FileNotFoundError(f"Cartella labels non trovata: {self.ann_dir}")

        self.img_paths = sorted(
            glob.glob(os.path.join(self.img_dir, "*.jpg")) +
            glob.glob(os.path.join(self.img_dir, "*.jpeg")) +
            glob.glob(os.path.join(self.img_dir, "*.png"))
        )
        if len(self.img_paths) == 0:
            raise RuntimeError(f"Nessuna immagine trovata in {self.img_dir}")

    def __len__(self) -> int:
        return len(self.img_paths)

    # ---- helpers ----
    def _match_npy_for_image(self, img_path: str) -> str:
        """
        Trova il .npy corrispondente all'immagine.
        Prova: <base>.npy, rimuove 'IMG_', prova zfill 3/4... e cerca per glob.
        """
        base = os.path.splitext(os.path.basename(img_path))[0]  # e.g., IMG_002 o 002
        # 1) match diretto
        p = os.path.join(self.ann_dir, f"{base}.npy")
        if os.path.exists(p):
            return p
        # 2) senza prefisso
        alt = base.replace("IMG_", "")
        for cand in (alt, alt.zfill(3), alt.zfill(4), alt.zfill(5)):
            p = os.path.join(self.ann_dir, f"{cand}.npy")
            if os.path.exists(p):
                return p
        # 3) glob rilassato
        matches = sorted(glob.glob(os.path.join(self.ann_dir, f"*{alt}*.npy")))
        if matches:
            return matches[0]
        raise FileNotFoundError(f".npy per {base} non trovato in {self.ann_dir}")

    # ---- main ----
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_path = self.img_paths[idx]
        base_name = os.path.splitext(os.path.basename(img_path))[0]

        # 1) immagine
        img = safe_load_image(img_path)  # [H,W,3] uint8

        # 2) density map (Shanghai .npy)
        npy_path = self._match_npy_for_image(img_path)
        density = np.load(npy_path)
        if density.ndim == 3:
            density = density[..., 0]
        density = density.astype(np.float32)

        # -> allinea subito density a img (richiesto da Albumentations)
        if density.shape != img.shape[:2]:
           density = resize_density_map_keep_count(density, img.shape[0], img.shape[1])
           
        # 3) transforms (stesse geo-trasformazioni su immagine e mask)
        if self.transform is not None:
            try:
                augmented = self.transform(image=img, mask=density)
                img_t = augmented["image"]
                density_t = augmented["mask"]
                # Se ToTensorV2 è stato usato, density_t è torch.Tensor; altrimenti numpy
                if isinstance(density_t, torch.Tensor):
                    density_np = density_t.cpu().numpy().astype(np.float32)
                else:
                    density_np = density_t.astype(np.float32)
            except TypeError:
                # nel fallback custom
                augmented = self.transform(image=img, mask=density)
                img_t = augmented["image"]
                density_np = augmented["mask"].numpy().astype(np.float32)
        else:
            # fallback manuale
            img_t = torch.from_numpy((img.astype(np.float32) / 255.0)).permute(2, 0, 1)
            density_np = density

        # 4) riallinea density a H,W esatti (safety) preservando il conteggio
        H, W = img_t.shape[-2], img_t.shape[-1]
        if density_np.shape != (H, W):
            density_np = resize_density_map_keep_count(density_np, H, W)

        # 5) target blockwise + π-head (senza costanti arbitrarie)
        bs = self.block_size  # es. 8 pixel
        block_counts = block_sum(density_np, bs, bs).astype(np.float32)  # [Hb,Wb]

        # Primo criterio (rigoroso): blocco vuoto se conteggio esattamente 0
        pi_gt = (block_counts == 0.0).astype(np.float32)

        # Fallback adattivo se non esistono blocchi a zero (capita con alcune pre-elab)
        if pi_gt.sum() == 0:
            # normalizza per pixel e stima soglia con Otsu (data-driven)
            per_pix = block_counts / float(bs * bs)
            thr = otsu_threshold(per_pix)
            pi_gt = (per_pix <= thr).astype(np.float32)

        sample = {
            "image": img_t,
            "gt_density": torch.from_numpy(density_np),
            "gt_block_counts": torch.from_numpy(block_counts),
            "gt_pi": torch.from_numpy(pi_gt),
            "base_name": base_name,
            "image_path": img_path,
            "label_path": npy_path,
        }
        return sample
        
def get_transforms(split: str = "train"):
    """Ritorna la pipeline di trasformazioni (lazy)."""
    return lazy_get_transforms(split)
