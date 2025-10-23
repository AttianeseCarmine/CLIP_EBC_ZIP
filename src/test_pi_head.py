# test_pi_head.py
# Uso:
#   python -m src.test_pi_head --image path/alla/tua.jpg
#   # opzionali:
#   # --adaptive (usa AdaptivePiHead)
#   # --hard-mask (usa maschera dura; default soft)

import os
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

# Import robusto: funziona sia con `python -m src.test_pi_head`
# sia con `python src/test_pi_head.py` se PYTHONPATH include src/
try:
    from .pi_head import PiHead, AdaptivePiHead
    from .visualize_pi_head import PiHeadVisualizer
except ImportError:
    from pi_head import PiHead, AdaptivePiHead
    from visualize_pi_head import PiHeadVisualizer


# ------------------------------
# Utils
# ------------------------------
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_image_rgb(path: str, target_size: int = 512):
    if not Path(path).exists():
        raise FileNotFoundError(f"Immagine non trovata: {path}")
    img = Image.open(path).convert("RGB")
    img = img.resize((target_size, target_size))
    img_np = np.array(img)  # [H,W,3] uint8
    img_t = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0  # [3,H,W] float
    return img_np, img_t.unsqueeze(0)  # (np per viz, torch per modello)


def tiny_backbone(in_ch=3, out_ch=256):
    # Feature H/8 x W/8, ~256 canali
    return nn.Sequential(
        nn.Conv2d(in_ch, 64, 3, stride=2, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 128, 3, stride=2, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(128, out_ch, 3, stride=2, padding=1),
        nn.ReLU(inplace=True),
    )


# ------------------------------
# Main
# ------------------------------
def run(args):
    device = get_device()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")

    # Carica immagine (OBBLIGATORIA)
    img_np, img_t = load_image_rgb(args.image, target_size=args.input_size)
    img_t = img_t.to(device)

    # Backbone leggero -> feature map
    backbone = tiny_backbone(out_ch=args.feature_channels).to(device)
    backbone.eval()
    with torch.no_grad():
        features = backbone(img_t)  # [1,C,H/8,W/8]

    # PiHead o AdaptivePiHead
    HeadCls = AdaptivePiHead if args.adaptive else PiHead
    pi_head = HeadCls(
        in_channels=args.feature_channels,
        hidden_channels=args.hidden_channels,
        num_layers=args.num_layers,
        threshold=args.threshold,
    ).to(device)
    pi_head.eval()

    # Forward (solo immagine reale: niente density_gt)
    with torch.no_grad():
        out = pi_head(
            features,
            return_mask=True,
            refine=True,
            #hard_mask=args.hard_mask,  # soft (default) o hard (se flag)
        )

    pi_map = out["pi"][0, 0].detach().cpu().numpy()
    mask = out["mask"][0, 0].detach().cpu().numpy()
    thr = out.get("threshold", None)
    sample_name = Path(args.image).stem

    # Visualizzazione e salvataggio
    vis = PiHeadVisualizer()
    save_path = out_dir / f"{sample_name}_pi_{stamp}.png"
    vis.visualize_prediction(
        image=img_np,
        pi_map=pi_map,
        mask=mask,
        density_gt=None,  # non disponibile su immagine reale
        save_path=str(save_path),
    )

    # Log
    print(f"[OK] Salvato: {save_path}")
    print(f"Device: {device}")
    print(f"Head: {'AdaptivePiHead' if args.adaptive else 'PiHead'}")
    print(f"hard_mask: {args.hard_mask}")
    print(f"threshold: {thr if thr is not None else args.threshold}")
    print(f"features: {tuple(features.shape)}")


def build_parser():
    p = argparse.ArgumentParser("Test PiHead / AdaptivePiHead su immagine reale")
    p.add_argument("--image", type=str, required=True, help="Percorso immagine RGB.")
    p.add_argument("--input-size", type=int, default=512, help="Resize lato immagine.")
    p.add_argument("--feature-channels", type=int, default=256, help="Canali feature dal backbone.")
    p.add_argument("--hidden-channels", type=int, default=256, help="Canali hidden del PiHead.")
    p.add_argument("--num-layers", type=int, default=2, help="Numero layer conv nel PiHead.")
    p.add_argument("--threshold", type=float, default=0.5, help="Soglia fissa (ignorata da Adaptive).")
    p.add_argument("--adaptive", action="store_true", help="Usa AdaptivePiHead.")
    p.add_argument("--hard-mask", action="store_true", help="Usa maschera dura (eval); default soft.")
    p.add_argument("--output-dir", type=str, default="outputs", help="Cartella output.")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    run(args)
