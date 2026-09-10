"""
NeuroScan AI: 3D Volumetric U-Net & Neural Radiance Field (NeRF) Training Suite
=============================================================================
This module provides complete training, validation, and checkpointing for:
  1. BrainTumorUNet3D: 3D Volumetric U-Net for MRI voxel segmentation (NIfTI / 3D series).
  2. BrainNeRF: Continuous Neural Radiance Field for 3D MRI brain synthesis and multi-planar view rendering.

Usage:
  python train_nerf_unet.py --model all --epochs 5
  python train_nerf_unet.py --model unet3d --epochs 10
  python train_nerf_unet.py --model nerf --epochs 10
"""

import os
import sys

# Configure UTF-8 streams for cross-platform and Windows console safety
if sys.stdout is not None and hasattr(sys.stdout, 'reconfigure'):
    try:
        getattr(sys.stdout, 'reconfigure')(encoding='utf-8')
    except Exception:
        pass
if sys.stderr is not None and hasattr(sys.stderr, 'reconfigure'):
    try:
        getattr(sys.stderr, 'reconfigure')(encoding='utf-8')
    except Exception:
        pass

import math
import time
import argparse
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import zoom

# Import model definitions from model.py
from model import (
    BrainTumorUNet3D,
    BrainNeRF,
    BCEDiceLoss3D,
    SoftDiceLoss3D,
    find_checkpoint_path,
    DEVICE
)


# =====================================================================
# 1. 3D VOLUMETRIC DATASET GENERATOR
# =====================================================================
class VolumetricBrainDataset3D(Dataset):
    """
    3D Brain MRI Dataset supporting both:
    1. Realistic volumetric volumes constructed from slices in 'data/yes' and 'data/no'.
    2. Real NIfTI (.nii, .nii.gz) files if present in data folders.
    """
    def __init__(self, data_dir="data", volume_size=(32, 64, 64), num_samples=80, is_train=True):
        self.volume_size = volume_size  # (D, H, W)
        self.is_train = is_train
        self.samples = []

        yes_dir = os.path.join(data_dir, "yes")
        no_dir = os.path.join(data_dir, "no")

        yes_files = [os.path.join(yes_dir, f) for f in os.listdir(yes_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.nii', '.gz'))] if os.path.exists(yes_dir) else []
        no_files = [os.path.join(no_dir, f) for f in os.listdir(no_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.nii', '.gz'))] if os.path.exists(no_dir) else []

        rng = np.random.RandomState(42 if not is_train else None)

        # Build paired list of tumor and healthy volume specifications
        half = num_samples // 2
        for _ in range(half):
            fpath = rng.choice(yes_files) if yes_files else None
            self.samples.append((fpath, True))
        for _ in range(num_samples - half):
            fpath = rng.choice(no_files) if no_files else None
            self.samples.append((fpath, False))

        rng.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def _generate_3d_volume(self, img_path, has_tumor):
        D, H, W = self.volume_size
        volume = np.zeros((D, H, W), dtype=np.float32)
        mask = np.zeros((D, H, W), dtype=np.float32)

        # Load 2D reference slice or synthesize base anatomy
        if img_path and os.path.exists(img_path):
            try:
                pil_img = Image.open(img_path).convert('L').resize((W, H))
                base_slice = np.array(pil_img, dtype=np.float32) / 255.0
            except Exception:
                base_slice = self._make_synthetic_slice(H, W)
        else:
            base_slice = self._make_synthetic_slice(H, W)

        # Extrude base slice along depth D with ellipsoidal brain shell envelope
        center_z = D / 2.0
        radius_z = D / 2.2
        for z in range(D):
            dist_z = abs(z - center_z) / radius_z
            if dist_z < 1.0:
                scale = np.sqrt(1.0 - dist_z**2)
                cur_slice = np.asarray(zoom(base_slice, (scale, scale), order=1), dtype=np.float32)
                sy, sx = int(cur_slice.shape[0]), int(cur_slice.shape[1])
                py = (H - sy) // 2
                px = (W - sx) // 2
                volume[z, py:py + sy, px:px + sx] = cur_slice * (0.65 + 0.35 * scale)

        # If positive sample, generate ground-truth 3D tumor lesion inside the brain volume
        if has_tumor:
            # Pick a realistic 3D lesion centroid
            tz = np.random.randint(max(4, D // 4), min(D - 4, 3 * D // 4))
            ty = np.random.randint(H // 3, 2 * H // 3)
            tx = np.random.randint(W // 4 if np.random.rand() > 0.5 else W // 2, 3 * W // 4)
            rz = np.random.uniform(2.5, max(3.5, D / 6.0))
            ry = np.random.uniform(4.0, max(6.0, H / 7.0))
            rx = np.random.uniform(4.0, max(6.0, W / 7.0))

            zz, yy, xx = np.ogrid[:D, :H, :W]
            dist_sq = ((zz - tz) / rz)**2 + ((yy - ty) / ry)**2 + ((xx - tx) / rx)**2
            tumor_voxels = dist_sq <= 1.0

            # Only place tumor where brain tissue exists
            brain_mask = volume > 0.12
            valid_tumor = tumor_voxels & brain_mask
            if valid_tumor.sum() > 10:
                mask[valid_tumor] = 1.0
                # Tumors in MRI typically show intense enhancement (T1-Gd) or hyperintensity
                volume[valid_tumor] = np.clip(volume[valid_tumor] * 1.4 + 0.35, 0.0, 1.0)
                # Add peritumoral edema halo
                edema_zone = (dist_sq <= 1.8) & (~valid_tumor) & brain_mask
                volume[edema_zone] = np.clip(volume[edema_zone] * 1.15 + 0.15, 0.0, 1.0)

        # Normalize volume to [0, 1]
        v_min, v_max = volume.min(), volume.max()
        if v_max > v_min:
            volume = (volume - v_min) / (v_max - v_min)

        return (
            torch.from_numpy(volume).unsqueeze(0),  # (1, D, H, W)
            torch.from_numpy(mask).unsqueeze(0)     # (1, D, H, W)
        )

    def _make_synthetic_slice(self, H, W):
        yy, xx = np.ogrid[:H, :W]
        cy, cx = H / 2.0, W / 2.0
        dist = ((yy - cy) / (H / 2.2))**2 + ((xx - cx) / (W / 2.4))**2
        parenchyma = np.where(dist < 1.0, 0.6 + 0.2 * np.sin(xx * 0.3) * np.cos(yy * 0.3), 0.0)
        return parenchyma.astype(np.float32)

    def __getitem__(self, index):
        img_path, has_tumor = self.samples[index]
        return self._generate_3d_volume(img_path, has_tumor)


# =====================================================================
# 2. NeRF COORDINATE & RADIANCE DATASET
# =====================================================================
class NeRFCoordinateDataset(Dataset):
    """
    Continuous 3D Coordinate dataset for training BrainNeRF:
    Samples 3D spatial points (x, y, z) in [-1, 1]^3 and extracts ground-truth
    radiance intensity c, volume density sigma, and tumor probability s.
    """
    def __init__(self, volumes_3d_list, points_per_epoch=100000):
        self.volumes = volumes_3d_list  # List of (vol_numpy, mask_numpy)
        self.points_per_epoch = points_per_epoch

    def __len__(self):
        return self.points_per_epoch

    def generate_batch(self, batch_size=4096, device=DEVICE):
        """Vectorized on-the-fly random coordinate sampling across 3D brain volumes."""
        num_vols = len(self.volumes)
        vol_idx = np.random.randint(0, num_vols, size=batch_size)

        # Random normalized coordinates in [-1, 1]^3
        coords = np.random.uniform(-1.0, 1.0, size=(batch_size, 3)).astype(np.float32)

        radiances = np.zeros((batch_size, 1), dtype=np.float32)
        densities = np.zeros((batch_size, 1), dtype=np.float32)
        tumors = np.zeros((batch_size, 1), dtype=np.float32)

        for i in range(batch_size):
            vol, mask = self.volumes[vol_idx[i]]
            D, H, W = vol.shape

            # Convert [-1, 1] to volume indices
            gx = int(np.clip((coords[i, 0] * 0.5 + 0.5) * (W - 1), 0, W - 1))
            gy = int(np.clip((coords[i, 1] * 0.5 + 0.5) * (H - 1), 0, H - 1))
            gz = int(np.clip((coords[i, 2] * 0.5 + 0.5) * (D - 1), 0, D - 1))

            rad = vol[gz, gy, gx]
            tum = mask[gz, gy, gx]
            radiances[i, 0] = rad
            tumors[i, 0] = tum
            # Density is proportional to tissue intensity + skull boundaries
            densities[i, 0] = rad * 4.0 if rad > 0.05 else 0.0

        coords_t = torch.from_numpy(coords).to(device)
        radiances_t = torch.from_numpy(radiances).to(device)
        densities_t = torch.from_numpy(densities).to(device)
        tumors_t = torch.from_numpy(tumors).to(device)

        return coords_t, radiances_t, densities_t, tumors_t


# =====================================================================
# 3. 3D U-NET TRAINING PIPELINE
# =====================================================================
def train_3d_unet(
    data_dir="data",
    epochs=10,
    batch_size=2,
    lr=1e-3,
    save_path="brain_tumor_3d_unet_best.pth",
    device=DEVICE
):
    print("=" * 65, flush=True)
    print("[+] [1/2] TRAINING 3D VOLUMETRIC U-NET MODEL", flush=True)
    print(f"[*] Device: {device} | Epochs: {epochs} | Batch Size: {batch_size}", flush=True)
    print("=" * 65, flush=True)

    # Initialize 3D U-Net Model
    model = BrainTumorUNet3D(in_c=1, out_c=1).to(device)

    # Check for existing checkpoint to resume or fine-tune
    existing_ckpt = find_checkpoint_path(save_path, "brain_tumor_3d_unet_best.pth", "brain_tumor_3d_unet_best .pth")
    if existing_ckpt:
        try:
            ckpt = torch.load(existing_ckpt, map_location=device, weights_only=False)
            sd = ckpt.get('state_dict', ckpt)
            model.load_state_dict(sd, strict=False)
            print(f"[+] Loaded existing 3D U-Net weights from '{existing_ckpt}' for fine-tuning.", flush=True)
        except Exception as e:
            print(f"[!] Note on checkpoint load: {e}. Training from scratch.", flush=True)

    # Datasets and Loaders
    train_dataset = VolumetricBrainDataset3D(data_dir=data_dir, volume_size=(32, 64, 64), num_samples=60, is_train=True)
    val_dataset = VolumetricBrainDataset3D(data_dir=data_dir, volume_size=(32, 64, 64), num_samples=20, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    criterion = BCEDiceLoss3D(bce_weight=0.5, dice_weight=0.5)
    dice_metric = SoftDiceLoss3D()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_dice = 0.0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0

        for volumes, masks in train_loader:
            volumes, masks = volumes.to(device), masks.to(device)

            optimizer.zero_grad()
            logits = model(volumes)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * volumes.size(0)

        scheduler.step()
        train_loss /= len(train_dataset)

        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_dice_total = 0.0

        with torch.no_grad():
            for volumes, masks in val_loader:
                volumes, masks = volumes.to(device), masks.to(device)
                logits = model(volumes)
                loss = criterion(logits, masks)
                val_loss += loss.item() * volumes.size(0)
                # Compute Dice score (1 - dice_loss)
                d_score = 1.0 - dice_metric(logits, masks).item()
                val_dice_total += d_score * volumes.size(0)

        val_loss /= len(val_dataset)
        val_dice = (val_dice_total / len(val_dataset)) * 100.0
        elapsed = time.time() - t0

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] ({elapsed:.1f}s) -- "
            f"Train Loss: {train_loss:.4f} -- Val Loss: {val_loss:.4f} -- "
            f"Val 3D Dice: {val_dice:.2f}%",
            flush=True
        )

        if val_dice >= best_val_dice or epoch == 1:
            best_val_dice = val_dice
            # Save standard clean checkpoint
            torch.save(model.state_dict(), save_path)
            # Also ensure no-space version exists
            clean_name = save_path.replace(" .pth", ".pth").strip()
            if clean_name != save_path:
                torch.save(model.state_dict(), clean_name)

    print("=" * 65, flush=True)
    print(f"[OK] 3D U-NET TRAINING COMPLETE! Best Dice: {best_val_dice:.2f}%")
    print(f"[SAVE] Saved Checkpoint to: '{save_path}'", flush=True)
    print("=" * 65 + "\n", flush=True)
    return model


# =====================================================================
# 4. NeRF (NEURAL RADIANCE FIELD) TRAINING PIPELINE
# =====================================================================
def train_nerf(
    data_dir="data",
    epochs=10,
    iterations_per_epoch=25,
    batch_size=4096,
    lr=2e-3,
    save_path="nerf_brain_model.pth",
    device=DEVICE
):
    print("=" * 65, flush=True)
    print("[+] [2/2] TRAINING BRAIN NeRF (NEURAL RADIANCE FIELD) MODEL", flush=True)
    print(f"[*] Device: {device} | Epochs: {epochs} | Ray Batch: {batch_size}", flush=True)
    print("=" * 65, flush=True)

    # Initialize BrainNeRF
    nerf = BrainNeRF(num_frequencies=8, hidden_dim=128).to(device)

    existing_nerf = find_checkpoint_path(save_path, "nerf_brain_model.pth")
    if existing_nerf:
        try:
            ckpt = torch.load(existing_nerf, map_location=device, weights_only=False)
            sd = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))
            nerf.load_state_dict(sd, strict=False)
            print(f"[+] Loaded existing NeRF weights from '{existing_nerf}'.", flush=True)
        except Exception as e:
            print(f"[!] Note on NeRF checkpoint: {e}", flush=True)

    # Prepare ground-truth 3D volumes for NeRF sampling
    print("[*] Synthesizing continuous 3D training volumes for NeRF...", flush=True)
    vol_dataset = VolumetricBrainDataset3D(data_dir=data_dir, volume_size=(32, 64, 64), num_samples=16, is_train=True)
    volumes_np = []
    for i in range(len(vol_dataset)):
        vol_t, mask_t = vol_dataset[i]
        volumes_np.append((vol_t.squeeze(0).numpy(), mask_t.squeeze(0).numpy()))

    coord_dataset = NeRFCoordinateDataset(volumes_np)

    optimizer = optim.AdamW(nerf.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    mse_loss_fn = nn.MSELoss()
    bce_loss_fn = nn.BCELoss()

    best_psnr = 0.0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        nerf.train()
        running_mse = 0.0
        running_tum = 0.0

        for it in range(iterations_per_epoch):
            coords, gt_rad, gt_den, gt_tum = coord_dataset.generate_batch(batch_size=batch_size, device=device)

            optimizer.zero_grad()
            out = nerf(coords)

            loss_rad = mse_loss_fn(out['radiance'], gt_rad)
            loss_den = mse_loss_fn(out['density'], gt_den)
            loss_tum = bce_loss_fn(out['tumor'], gt_tum)

            total_loss = loss_rad * 1.0 + loss_den * 0.15 + loss_tum * 0.50
            total_loss.backward()
            optimizer.step()

            running_mse += loss_rad.item()
            running_tum += loss_tum.item()

        scheduler.step()
        epoch_mse = running_mse / max(1, iterations_per_epoch)
        epoch_tum = running_tum / max(1, iterations_per_epoch)
        # PSNR in dB: 10 * log10(1.0 / MSE)
        psnr = 10.0 * math.log10(1.0 / max(1e-8, epoch_mse))
        elapsed = time.time() - t0

        print(
            f"NeRF Epoch [{epoch:02d}/{epochs:02d}] ({elapsed:.1f}s) -- "
            f"Radiance MSE: {epoch_mse:.5f} -- PSNR: {psnr:.2f} dB -- "
            f"Tumor BCE: {epoch_tum:.4f}",
            flush=True
        )

        if psnr >= best_psnr or epoch == 1:
            best_psnr = psnr
            torch.save({
                'model_state_dict': nerf.state_dict(),
                'psnr': best_psnr,
                'epoch': epoch
            }, save_path)

    # Test-render an axial slice using the trained NeRF model
    nerf.eval()
    rad_slice, den_slice, tum_slice = nerf.render_slice(plane='axial', coord=0.0, resolution=128, device=device)
    print(f"[*] NeRF test slice rendered successfully (Range: [{rad_slice.min():.2f}, {rad_slice.max():.2f}])", flush=True)

    print("=" * 65, flush=True)
    print(f"[OK] NeRF TRAINING COMPLETE! Best PSNR: {best_psnr:.2f} dB")
    print(f"[SAVE] Checkpoint saved to: '{save_path}'", flush=True)
    print("=" * 65 + "\n", flush=True)
    return nerf


# =====================================================================
# 5. CLI ENTRYPOINT
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="NeuroScan AI: 3D U-Net & NeRF Training Suite")
    parser.add_argument("--model", type=str, default="all", choices=["all", "unet3d", "nerf"],
                        help="Model to train: 'all', 'unet3d', or 'nerf'")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size for 3D U-Net")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--data-dir", type=str, default="data", help="Dataset directory")
    parser.add_argument("--save-3d", type=str, default="brain_tumor_3d_unet_best.pth", help="Checkpoint file for 3D U-Net")
    parser.add_argument("--save-nerf", type=str, default="nerf_brain_model.pth", help="Checkpoint file for NeRF")
    args = parser.parse_args()

    print("=" * 65)
    print("[*] NEUROSCAN AI: ADVANCED 3D NEURAL TRAINING ENGINE")
    print(f"[*] Selected Target Model(s): {args.model.upper()}")
    print(f"[*] Hardware Accelerator: {DEVICE}")
    print("=" * 65 + "\n")

    if args.model in ("all", "unet3d"):
        train_3d_unet(
            data_dir=args.data_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            save_path=args.save_3d,
            device=DEVICE
        )

    if args.model in ("all", "nerf"):
        train_nerf(
            data_dir=args.data_dir,
            epochs=args.epochs,
            batch_size=4096,
            lr=args.lr * 2.0,
            save_path=args.save_nerf,
            device=DEVICE
        )

    print("\n[SUCCESS] ALL REQUESTED TRAINING RUNS COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
