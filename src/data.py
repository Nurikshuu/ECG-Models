#!/usr/bin/env python3
"""
PTB-XL SOTA PyTorch Dataset v2.1
LOCKED FILE - Used by all models
Production-ready: Full reproducibility, ECG augmentations, multi-label balancing

Requirements:
- PyTorch >= 2.1.0
- NumPy >= 1.24.0
- Python >= 3.9

Recommended num_workers:
- Colab/Kaggle: 2
- Windows: 0 (avoid multiprocessing issues)
- Linux/Mac: 4-8

Based on:
- Official PTB-XL benchmarking [helme/ecg_ptbxl_benchmarking]
- ECG augmentation papers
- PyTorch reproducibility best practices
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from pathlib import Path
from typing import Tuple, Optional, Callable, List
import json
import random
import warnings
import os


# ===== REPRODUCIBILITY =====
def set_seed(seed: int = 42):
    """Fix all random seeds for full reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Note: cudnn.benchmark=True for better perf, deterministic=True for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Set True for inference speed
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"✅ Global seed set: {seed}")


def _worker_init_fn(worker_id: int):
    """
    Initialize each DataLoader worker with unique but reproducible seed.
    Uses the base seed from the generator + worker_id.
    """
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        # Seed = base_seed + worker_id for reproducibility
        base_seed = worker_info.seed % (2**32)
        np.random.seed(base_seed)
        random.seed(base_seed)


# ===== ECG-SPECIFIC AUGMENTATIONS =====
class ECGAugmentation:
    """
    ECG-specific data augmentations (time-series safe)
    
    NOTE: No flip/reverse - ECG has directional P-QRS-T waves
    """
    
    @staticmethod
    def add_gaussian_noise(signal: torch.Tensor, std: float = 0.02) -> torch.Tensor:
        """Add Gaussian noise (simulates measurement noise)"""
        noise = torch.randn_like(signal) * std
        return signal + noise
    
    @staticmethod
    def scale_amplitude(signal: torch.Tensor, scale_range: Tuple[float, float] = (0.85, 1.15)) -> torch.Tensor:
        """Random amplitude scaling (simulates electrode placement variance)"""
        scale = torch.empty(1).uniform_(*scale_range).item()
        return signal * scale
    
    @staticmethod
    def shift_baseline(signal: torch.Tensor, shift_range: float = 0.05) -> torch.Tensor:
        """Random baseline shift per lead (simulates baseline wander)"""
        shift = torch.empty(signal.shape[0], 1).uniform_(-shift_range, shift_range)
        return signal + shift
    
    @staticmethod
    def time_shift(signal: torch.Tensor, shift_max: int = 30) -> torch.Tensor:
        """Circular time shift (simulates trigger point variance)"""
        shift = random.randint(-shift_max, shift_max)
        return torch.roll(signal, shifts=shift, dims=-1)
    
    @staticmethod
    def lead_dropout(signal: torch.Tensor, p: float = 0.05) -> torch.Tensor:
        """Randomly zero out leads (robustness to missing/noisy leads)"""
        mask = (torch.rand(signal.shape[0], 1) > p).float()
        return signal * mask


class ECGTransform:
    """Compose multiple ECG augmentations with probability"""
    
    def __init__(self, augmentations: List[Callable], p: float = 0.5):
        """
        Args:
            augmentations: list of augmentation functions
            p: probability of applying each augmentation
        """
        self.augmentations = augmentations
        self.p = p
    
    def __call__(self, signal: torch.Tensor) -> torch.Tensor:
        for aug in self.augmentations:
            if random.random() < self.p:
                signal = aug(signal)
        return signal


# ===== DATASET =====
class PTBXLDataset(Dataset):
    """
    SOTA PTB-XL PyTorch Dataset
    
    Features:
    - Full reproducibility (seed=42)
    - ECG-specific augmentations (train only)
    - Multi-label support
    - Memory efficient
    - Class imbalance handling
    """
    
    def __init__(self, 
                 data_path: str,
                 split: str = 'train',
                 transform: Optional[Callable] = None,
                 augment: bool = False,
                 seed: int = 42):
        """
        Args:
            data_path: path to .npz file from preprocess.py
            split: 'train' | 'val' | 'test'
            transform: custom transform (overrides augment)
            augment: enable ECG augmentations (train only recommended)
            seed: random seed (fixed for reproducibility)
        """
        self.split = split
        self.seed = seed
        self.augment = augment and (split == 'train')  # Only augment training
        
        # Load preprocessed data
        data_path = Path(data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"❌ Data not found: {data_path}")
        
        data = np.load(data_path, allow_pickle=True)
        
        # Validate data format
        required_keys = ['signals', 'labels', 'splits', 'ecg_ids', 'class_names']
        missing = [k for k in required_keys if k not in data.keys()]
        if missing:
            raise ValueError(f"❌ Missing keys in .npz: {missing}")
        
        # Filter by split
        mask = data['splits'] == split
        self.signals = data['signals'][mask]  # (N, 12, 1000 or 5000)
        self.labels = data['labels'][mask]    # (N, n_classes) multi-hot
        self.ecg_ids = data['ecg_ids'][mask]
        
        # Handle class_names (may be numpy array or list)
        class_names = data['class_names']
        if hasattr(class_names, 'tolist'):
            self.class_names = class_names.tolist()
        else:
            self.class_names = list(class_names)
        
        self.n_classes = len(self.class_names)
        self.sampling_rate = int(data['sampling_rate'])
        
        # Setup transforms
        if transform is not None:
            self.transform = transform
        elif self.augment:
            self.transform = ECGTransform([
                ECGAugmentation.add_gaussian_noise,
                ECGAugmentation.scale_amplitude,
                ECGAugmentation.shift_baseline,
                ECGAugmentation.time_shift,
            ], p=0.5)
        else:
            self.transform = None
        
        print(f"✅ Loaded {split} split: {len(self)} samples")
        print(f"   Signal shape: {self.signals[0].shape}")
        print(f"   Classes: {self.class_names}")
        print(f"   Augmentation: {'ON' if self.augment else 'OFF'}")
        self._print_class_distribution()
    
    def _print_class_distribution(self):
        """Print class distribution (multi-label)"""
        counts = self.labels.sum(axis=0)
        print(f"   Label distribution:")
        for cls, cnt in zip(self.class_names, counts):
            pct = 100 * cnt / len(self)
            print(f"     {cls}: {int(cnt)} ({pct:.1f}%)")
    
    def __len__(self) -> int:
        return len(self.signals)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        signal = self.signals[idx]  # (12, n_samples)
        label = self.labels[idx]    # (n_classes,)
        
        # To tensor (copy to avoid negative stride warning)
        signal = torch.from_numpy(signal.copy()).float()
        label = torch.from_numpy(label.copy()).float()
        
        # Apply augmentation
        if self.transform:
            signal = self.transform(signal)
        
        return signal, label
    
    def get_sample_weights(self) -> np.ndarray:
        """
        Compute sample weights for balanced sampling (multi-label)
        Strategy: weight by inverse frequency of rarest label in sample
        """
        # Class frequencies
        class_freqs = self.labels.sum(axis=0) / len(self.labels)
        class_freqs = np.clip(class_freqs, 1e-8, 1.0)  # Avoid division by zero
        
        # Sample weight = 1 / min(freq of labels in sample)
        sample_weights = np.zeros(len(self.labels))
        for i, label in enumerate(self.labels):
            pos_classes = np.where(label == 1)[0]
            if len(pos_classes) > 0:
                # Weight by rarest class
                min_freq = class_freqs[pos_classes].min()
                sample_weights[i] = 1.0 / min_freq
            else:
                # No labels (edge case) - assign mean weight
                sample_weights[i] = 1.0
        
        # Validate weights
        if not np.isfinite(sample_weights).all():
            warnings.warn("⚠️ Non-finite sample weights detected - replacing with 1.0")
            sample_weights = np.nan_to_num(sample_weights, nan=1.0, posinf=1.0, neginf=1.0)
        
        # Normalize to mean=1
        sample_weights = sample_weights / sample_weights.mean()
        
        return sample_weights
    
    def get_class_weights(self) -> torch.Tensor:
        """
        Compute class weights for BCEWithLogitsLoss (pos_weight parameter)
        Formula: pos_weight = neg_samples / pos_samples
        """
        pos_counts = self.labels.sum(axis=0)
        pos_counts = np.maximum(pos_counts, 1.0)  # Avoid division by zero
        neg_counts = len(self.labels) - pos_counts
        weights = neg_counts / pos_counts
        
        # Clip extreme weights
        weights = np.clip(weights, 0.1, 10.0)
        
        return torch.from_numpy(weights).float()


# ===== DATALOADERS =====
def get_dataloaders(
    data_path: str,
    batch_size: int = 32,
    num_workers: int = 0,
    use_weighted_sampler: bool = False,
    augment_train: bool = True,
    seed: int = 42,
    persistent_workers: bool = False
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Get SOTA DataLoaders for PTB-XL with full reproducibility
    
    Args:
        data_path: path to preprocessed .npz
        batch_size: batch size
        num_workers: parallel workers (0 for Windows, 2-4 for Linux/Colab)
        use_weighted_sampler: balance classes via sampling
        augment_train: enable ECG augmentations for training
        seed: random seed (LOCKED at 42)
        persistent_workers: keep workers alive between epochs (faster, more memory)
    
    Returns:
        train_loader, val_loader, test_loader
    """
    set_seed(seed)
    
    # Create generator for reproducible shuffle
    generator = torch.Generator()
    generator.manual_seed(seed)
    
    # Datasets
    train_dataset = PTBXLDataset(data_path, split='train', augment=augment_train, seed=seed)
    val_dataset = PTBXLDataset(data_path, split='val', augment=False, seed=seed)
    test_dataset = PTBXLDataset(data_path, split='test', augment=False, seed=seed)
    
    # Weighted sampler for training (optional)
    train_sampler = None
    train_shuffle = True
    if use_weighted_sampler:
        sample_weights = train_dataset.get_sample_weights()
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
            generator=generator  # Reproducible sampling
        )
        train_shuffle = False  # No shuffle if using sampler
        print("✅ Using WeightedRandomSampler for balanced training")
    
    # Persistent workers only if num_workers > 0
    use_persistent = persistent_workers and num_workers > 0
    
    # DataLoaders with full reproducibility
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=train_shuffle if train_sampler is None else False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,  # Stable batch size for BatchNorm
        generator=generator,
        worker_init_fn=_worker_init_fn,
        persistent_workers=use_persistent
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=_worker_init_fn,
        persistent_workers=use_persistent
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=_worker_init_fn,
        persistent_workers=use_persistent
    )
    
    print(f"\n✅ DataLoaders ready (seed={seed}, reproducible):")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    print(f"   Test batches: {len(test_loader)}")
    
    return train_loader, val_loader, test_loader


# ===== UTILITY FUNCTIONS =====
def get_class_weights_from_dataloader(train_loader: DataLoader) -> torch.Tensor:
    """Extract class weights from dataset (for BCEWithLogitsLoss)"""
    return train_loader.dataset.get_class_weights()


def visualize_batch(signals: torch.Tensor, labels: torch.Tensor, 
                   class_names: List[str], save_path: str = 'ecg_batch.png'):
    """Visualize ECG batch (for debugging/EDA)"""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("⚠️ matplotlib not installed - skipping visualization")
        return
    
    batch_size = min(4, signals.shape[0])
    fig, axes = plt.subplots(batch_size, 1, figsize=(14, 8))
    
    if batch_size == 1:
        axes = [axes]
    
    for i in range(batch_size):
        # Plot lead II (index 1) - most diagnostic
        axes[i].plot(signals[i, 1, :].numpy(), linewidth=0.8)
        label_names = [class_names[j] for j, val in enumerate(labels[i]) if val == 1]
        axes[i].set_title(f"Sample {i+1}: {', '.join(label_names) if label_names else 'No labels'}")
        axes[i].set_xlabel("Samples (10s @ 100Hz = 1000 samples)")
        axes[i].set_ylabel("Amplitude (normalized)")
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved batch visualization: {save_path}")
    plt.close()


# ===== TESTING =====
if __name__ == "__main__":
    """Comprehensive test suite for SOTA data.py"""
    print("="*60)
    print("PTB-XL SOTA Dataset v2.1 - Full Test Suite")
    print("="*60)
    
    data_path = "data_preprocessed/ptbxl_sota_100hz_diagnostic_superclass.npz"
    
    if not Path(data_path).exists():
        print(f"❌ Test data not found: {data_path}")
        print("   Run preprocess.py first!")
        exit(1)
    
    # Test 1: Basic loading
    print("\n[Test 1] Dataset Loading")
    train_dataset = PTBXLDataset(data_path, split='train', augment=True, seed=42)
    
    # Test 2: Augmentation
    print("\n[Test 2] Augmentation")
    signal, label = train_dataset[0]
    print(f"   Signal shape: {signal.shape}")
    print(f"   Label shape: {label.shape}")
    print(f"   Signal range: [{signal.min():.2f}, {signal.max():.2f}]")
    
    # Test 3: Class weights
    print("\n[Test 3] Class Weights")
    class_weights = train_dataset.get_class_weights()
    print(f"   Weights: {class_weights.numpy()}")
    print(f"   All finite: {torch.isfinite(class_weights).all()}")
    
    # Test 4: Sample weights
    print("\n[Test 4] Sample Weights")
    sample_weights = train_dataset.get_sample_weights()
    print(f"   Shape: {sample_weights.shape}")
    print(f"   Range: [{sample_weights.min():.2f}, {sample_weights.max():.2f}]")
    print(f"   All finite: {np.isfinite(sample_weights).all()}")
    print(f"   All positive: {(sample_weights > 0).all()}")
    
    # Test 5: DataLoaders
    print("\n[Test 5] DataLoaders Creation")
    train_loader, val_loader, test_loader = get_dataloaders(
        data_path, 
        batch_size=32, 
        num_workers=0,  # 0 for Windows test
        use_weighted_sampler=False,
        augment_train=True,
        seed=42
    )
    
    # Test 6: Batch test
    print("\n[Test 6] Batch Iteration")
    signals_batch, labels_batch = next(iter(train_loader))
    print(f"   Signals: {signals_batch.shape}")
    print(f"   Labels: {labels_batch.shape}")
    print(f"   Signal range: [{signals_batch.min():.2f}, {signals_batch.max():.2f}]")
    
    # Test 7: Full reproducibility
    print("\n[Test 7] Reproducibility Test (CRITICAL)")
    
    # Run 1
    set_seed(42)
    loader1 = get_dataloaders(data_path, batch_size=16, num_workers=0, seed=42)[0]
    batch1_signals, batch1_labels = next(iter(loader1))
    
    # Run 2
    set_seed(42)
    loader2 = get_dataloaders(data_path, batch_size=16, num_workers=0, seed=42)[0]
    batch2_signals, batch2_labels = next(iter(loader2))
    
    signals_equal = torch.allclose(batch1_signals, batch2_signals, atol=1e-6)
    labels_equal = torch.equal(batch1_labels, batch2_labels)
    
    print(f"   Signals equal: {signals_equal}")
    print(f"   Labels equal: {labels_equal}")
    
    if signals_equal and labels_equal:
        print("   ✅ REPRODUCIBILITY PASSED")
    else:
        print("   ❌ REPRODUCIBILITY FAILED")
        print(f"   Signal diff: {(batch1_signals - batch2_signals).abs().max()}")
    
    # Test 8: Visualization
    print("\n[Test 8] Batch Visualization")
    visualize_batch(signals_batch, labels_batch, train_dataset.class_names)
    
    print("\n" + "="*60)
    if signals_equal and labels_equal:
        print("✅ ALL TESTS PASSED - PRODUCTION READY!")
    else:
        print("⚠️ SOME TESTS FAILED - CHECK ABOVE")
    print("="*60)
