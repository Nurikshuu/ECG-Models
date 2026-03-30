#!/usr/bin/env python3
"""
PTB-XL SOTA Preprocessing Pipeline v2.1
Production-ready: robust, scalable, reproducible
LOCKED FILE - All models ready

USAGE:
    python src/preprocess.py \
        --input_dir data_raw/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3 \
        --output_dir data_preprocessed \
        --sampling_rate 100 \
        --label_type diagnostic_superclass

OUTPUT:
    data_preprocessed/
    ├── ptbxl_sota_100hz_diagnostic_superclass.npz  (~800MB)
    ├── metadata_100hz.json
    └── errors_100hz.csv (if any)
"""

import argparse
import ast
import json
import logging
from pathlib import Path
from typing import Tuple, Dict, List
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
import wfdb
from tqdm import tqdm
from sklearn.preprocessing import MultiLabelBinarizer

# ===== CONFIG =====
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PTBXLPreprocessorSOTA:
    """Production-grade PTB-XL preprocessor"""
    
    # PTB-XL 5 Diagnostic Superclasses
    SUPERCLASS_MAPPING = {
        'NORM': 'NORM',  # Normal ECG
        'MI': 'MI',      # Myocardial Infarction
        'STTC': 'STTC',  # ST/T Change
        'CD': 'CD',      # Conduction Disturbance
        'HYP': 'HYP',    # Hypertrophy
    }
    
    def __init__(self, 
                 input_dir: Path,
                 output_dir: Path,
                 sampling_rate: int = 100,
                 bandpass: Tuple[float, float] = (0.5, 40.0),
                 label_type: str = 'diagnostic_superclass'):
        """
        Args:
            input_dir: path to PTB-XL extracted folder (contains ptbxl_database.csv)
            output_dir: data_preprocessed/
            sampling_rate: 100 or 500 Hz
            bandpass: (low, high) Hz cutoffs
            label_type: 'diagnostic_superclass' | 'diagnostic_class' | 'all'
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.fs = sampling_rate
        self.bandpass = bandpass
        self.label_type = label_type
        
        # Expected signal shape
        self.n_leads = 12
        self.duration_sec = 10
        self.n_samples = self.fs * self.duration_sec  # 1000 @ 100Hz, 5000 @ 500Hz
        
        # Validate input directory
        self._validate_input_dir()
        
        # Load metadata
        self.scp_statements = self._load_scp_statements()
        self.metadata_df = self._load_metadata()
        
        # Build SCP -> Superclass mapping
        self.scp_to_superclass = self._build_scp_mapping()
        
        # Setup class mapping
        self.mlb, self.class_names = self._setup_label_encoder()
        
        # Stats tracking
        self.stats = {'processed': 0, 'errors': 0, 'skipped': 0}
        self.error_log = []
    
    def _validate_input_dir(self):
        """Validate input directory structure"""
        required_files = ['ptbxl_database.csv', 'scp_statements.csv']
        required_dirs = ['records100' if self.fs == 100 else 'records500']
        
        for f in required_files:
            if not (self.input_dir / f).exists():
                raise FileNotFoundError(f"❌ Missing: {self.input_dir / f}")
        
        for d in required_dirs:
            if not (self.input_dir / d).exists():
                raise FileNotFoundError(f"❌ Missing directory: {self.input_dir / d}")
        
        logger.info(f"✅ Input directory validated: {self.input_dir}")
    
    def _load_scp_statements(self) -> pd.DataFrame:
        """Load and validate SCP statements"""
        csv_path = self.input_dir / 'scp_statements.csv'
        df = pd.read_csv(csv_path, index_col=0)  # First column is SCP code
        
        logger.info(f"✅ Loaded {len(df)} SCP statements")
        logger.info(f"   Columns: {df.columns.tolist()}")
        return df
    
    def _load_metadata(self) -> pd.DataFrame:
        """Load and validate metadata"""
        csv_path = self.input_dir / 'ptbxl_database.csv'
        df = pd.read_csv(csv_path)
        
        # Validate columns
        required = ['ecg_id', 'filename_lr', 'filename_hr', 'scp_codes', 'strat_fold']
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"❌ Missing columns in metadata: {missing}")
        
        # Add split column based on official PTB-XL recommendation
        fold_map = {i: 'train' for i in range(1, 9)}  # Folds 1-8: train
        fold_map.update({9: 'val', 10: 'test'})       # Fold 9: val, Fold 10: test
        df['split'] = df['strat_fold'].map(fold_map)
        
        logger.info(f"✅ Loaded {len(df)} records metadata")
        logger.info(f"   Train: {sum(df.split=='train')}, Val: {sum(df.split=='val')}, Test: {sum(df.split=='test')}")
        
        return df
    
    def _build_scp_mapping(self) -> Dict[str, str]:
        """Build SCP code -> diagnostic_class mapping"""
        mapping = {}
        for scp_code in self.scp_statements.index:
            row = self.scp_statements.loc[scp_code]
            if pd.notna(row.get('diagnostic_class')):
                mapping[scp_code] = row['diagnostic_class']
        
        logger.info(f"✅ SCP mapping: {len(mapping)} diagnostic codes")
        return mapping
    
    def _setup_label_encoder(self) -> Tuple[MultiLabelBinarizer, List[str]]:
        """Setup multi-label encoder with fixed class order"""
        if self.label_type == 'diagnostic_superclass':
            # 5 main classes: NORM, MI, STTC, CD, HYP
            classes = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
        elif self.label_type == 'diagnostic_class':
            # All diagnostic classes from SCP
            classes = sorted(self.scp_statements['diagnostic_class'].dropna().unique().tolist())
        else:
            # All SCP codes
            classes = sorted(self.scp_statements.index.tolist())
        
        mlb = MultiLabelBinarizer(classes=classes)
        mlb.fit([classes])  # Fit once with all classes
        
        logger.info(f"✅ Label encoder: {len(classes)} classes ({self.label_type})")
        logger.info(f"   Classes: {classes}")
        return mlb, classes
    
    def _parse_scp_codes(self, scp_str: str) -> List[str]:
        """Parse SCP codes from metadata string and map to target classes"""
        if pd.isna(scp_str) or not isinstance(scp_str, str):
            return []
        
        # Safe parsing: "{'NORM': 100.0, 'SR': 0.0}" -> dict
        try:
            scp_dict = ast.literal_eval(scp_str)
        except (ValueError, SyntaxError):
            logger.warning(f"Failed to parse SCP: {scp_str[:50]}...")
            return []
        
        # Map to target label type
        codes = set()
        for scp_code, likelihood in scp_dict.items():
            # Only use codes with likelihood > 0 (some papers use > 50)
            if likelihood <= 0:
                continue
            
            if self.label_type == 'diagnostic_superclass':
                # Map to 5 superclasses via diagnostic_class
                if scp_code in self.scp_to_superclass:
                    superclass = self.scp_to_superclass[scp_code]
                    if superclass in self.SUPERCLASS_MAPPING:
                        codes.add(superclass)
            elif self.label_type == 'diagnostic_class':
                if scp_code in self.scp_to_superclass:
                    codes.add(self.scp_to_superclass[scp_code])
            else:
                codes.add(scp_code)
        
        return list(codes)
    
    def _bandpass_filter(self, signal: np.ndarray) -> np.ndarray:
        """Butterworth bandpass filter"""
        nyquist = 0.5 * self.fs
        low = self.bandpass[0] / nyquist
        high = self.bandpass[1] / nyquist
        
        # Validate cutoffs
        if low <= 0 or high >= 1:
            logger.warning(f"Invalid bandpass {self.bandpass} for fs={self.fs}")
            return signal
        
        b, a = butter(4, [low, high], btype='band')
        return filtfilt(b, a, signal, axis=-1)
    
    def _preprocess_signal(self, signal: np.ndarray) -> np.ndarray:
        """Full preprocessing pipeline per record"""
        # Shape validation
        if signal.shape != (self.n_leads, self.n_samples):
            raise ValueError(f"Shape {signal.shape} != expected ({self.n_leads}, {self.n_samples})")
        
        # 1. Check for NaN/Inf
        if not np.isfinite(signal).all():
            signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
            logger.warning("Found NaN/Inf in signal - replaced with 0")
        
        # 2. Bandpass filter (0.5-40 Hz removes baseline wander + high freq noise)
        signal = self._bandpass_filter(signal)
        
        # 3. Baseline removal (subtract median per lead)
        baseline = np.median(signal, axis=-1, keepdims=True)
        signal = signal - baseline
        
        # 4. Z-score normalization per lead
        mean = signal.mean(axis=-1, keepdims=True)
        std = signal.std(axis=-1, keepdims=True) + 1e-8
        signal = (signal - mean) / std
        
        # 5. Clip outliers (±10 sigma)
        signal = np.clip(signal, -10, 10)
        
        return signal.astype(np.float32)
    
    def process_single_record(self, row: pd.Series) -> Dict:
        """Process one ECG record"""
        ecg_id = row['ecg_id']
        filename = row['filename_lr'] if self.fs == 100 else row['filename_hr']
        
        # Build path: input_dir/records100/00000/00001_lr
        record_path = self.input_dir / filename
        
        try:
            # Load WFDB (without extension)
            record = wfdb.rdrecord(str(record_path))
            raw_signal = record.p_signal  # (n_samples, n_leads)
            
            # Validate shape
            if raw_signal.shape[1] != self.n_leads:
                raise ValueError(f"Expected {self.n_leads} leads, got {raw_signal.shape[1]}")
            
            if raw_signal.shape[0] != self.n_samples:
                logger.warning(f"{ecg_id}: Expected {self.n_samples} samples, got {raw_signal.shape[0]} - padding/trimming")
                if raw_signal.shape[0] < self.n_samples:
                    pad = self.n_samples - raw_signal.shape[0]
                    raw_signal = np.pad(raw_signal, ((0, pad), (0, 0)), mode='edge')
                else:
                    raw_signal = raw_signal[:self.n_samples, :]
            
            # Transpose to (12, n_samples) for CNN
            signal = raw_signal.T
            
            # Preprocess
            signal = self._preprocess_signal(signal)
            
            # Parse labels
            scp_codes = self._parse_scp_codes(row['scp_codes'])
            label = self.mlb.transform([scp_codes])[0] if scp_codes else np.zeros(len(self.class_names))
            
            return {
                'ecg_id': ecg_id,
                'signal': signal,
                'label': label,
                'split': row['split'],
                'status': 'ok'
            }
            
        except Exception as e:
            logger.error(f"❌ {ecg_id}: {str(e)}")
            return {'ecg_id': ecg_id, 'status': 'error', 'error': str(e)}
    
    def run(self, max_records: int = None, n_workers: int = 1, save_per_record: bool = False):
        """Main preprocessing pipeline"""
        logger.info(f"🚀 Starting SOTA preprocessing: {self.fs}Hz, {self.label_type}")
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Select records
        df_subset = self.metadata_df.head(max_records) if max_records else self.metadata_df
        
        # Process
        results = []
        
        if n_workers > 1:
            # Parallel (careful with WFDB)
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = {executor.submit(self.process_single_record, row): idx 
                          for idx, row in df_subset.iterrows()}
                
                for future in tqdm(as_completed(futures), total=len(futures), desc="Preprocessing"):
                    result = future.result()
                    results.append(result)
        else:
            # Sequential (safer, recommended)
            for idx, row in tqdm(df_subset.iterrows(), total=len(df_subset), desc="Preprocessing"):
                result = self.process_single_record(row)
                results.append(result)
        
        # Filter successful
        ok_results = [r for r in results if r['status'] == 'ok']
        error_results = [r for r in results if r['status'] == 'error']
        
        logger.info(f"✅ Success: {len(ok_results)}, ❌ Errors: {len(error_results)}")
        
        # Save
        if save_per_record:
            self._save_per_record(ok_results)
        else:
            self._save_single_npz(ok_results)
        
        # Save metadata
        self._save_metadata(ok_results, error_results)
        
        return ok_results
    
    def _save_single_npz(self, results: List[Dict]):
        """Save all to one NPZ (default, efficient for training)"""
        signals = np.array([r['signal'] for r in results])
        labels = np.array([r['label'] for r in results])
        splits = np.array([r['split'] for r in results])
        ecg_ids = np.array([r['ecg_id'] for r in results])
        
        output_path = self.output_dir / f'ptbxl_sota_{self.fs}hz_{self.label_type}.npz'
        np.savez_compressed(
            output_path,
            signals=signals,
            labels=labels,
            splits=splits,
            ecg_ids=ecg_ids,
            class_names=self.class_names,
            sampling_rate=self.fs
        )
        
        size_gb = signals.nbytes / 1e9
        logger.info(f"💾 Saved: {output_path} ({size_gb:.2f} GB)")
        logger.info(f"   Shape: signals{signals.shape}, labels{labels.shape}")
        logger.info(f"   Classes: {self.class_names}")
    
    def _save_per_record(self, results: List[Dict]):
        """Save each record separately (scalable for huge datasets)"""
        records_dir = self.output_dir / 'records'
        records_dir.mkdir(exist_ok=True)
        
        for res in tqdm(results, desc="Saving records"):
            out_path = records_dir / f"{res['ecg_id']}.npz"
            np.savez_compressed(out_path, 
                               signal=res['signal'], 
                               label=res['label'],
                               split=res['split'])
        
        logger.info(f"💾 Saved {len(results)} records to {records_dir}/")
    
    def _save_metadata(self, ok_results: List[Dict], error_results: List[Dict]):
        """Save processing metadata + class mapping"""
        
        # Count labels per class
        label_counts = {}
        for cls_idx, cls_name in enumerate(self.class_names):
            count = sum(1 for r in ok_results if r['label'][cls_idx] == 1)
            label_counts[cls_name] = count
        
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'preprocessing_version': '2.1_SOTA',
            'sampling_rate': self.fs,
            'n_samples': self.n_samples,
            'n_leads': self.n_leads,
            'bandpass_hz': list(self.bandpass),
            'label_type': self.label_type,
            'n_classes': len(self.class_names),
            'class_names': self.class_names,
            'label_counts': label_counts,
            'n_processed': len(ok_results),
            'n_errors': len(error_results),
            'splits': {
                'train': sum(1 for r in ok_results if r['split'] == 'train'),
                'val': sum(1 for r in ok_results if r['split'] == 'val'),
                'test': sum(1 for r in ok_results if r['split'] == 'test')
            }
        }
        
        # Save JSON
        meta_path = self.output_dir / f'metadata_{self.fs}hz.json'
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Error log CSV
        if error_results:
            error_df = pd.DataFrame(error_results)
            error_path = self.output_dir / f'errors_{self.fs}hz.csv'
            error_df.to_csv(error_path, index=False)
            logger.info(f"📋 Error log: {error_path}")
        
        logger.info(f"📋 Metadata: {meta_path}")


def main():
    parser = argparse.ArgumentParser(description="PTB-XL SOTA Preprocessor v2.1")
    parser.add_argument('--input_dir', required=True, 
                       help='Path to PTB-XL folder (contains ptbxl_database.csv)')
    parser.add_argument('--output_dir', required=True, 
                       help='Output directory for preprocessed data')
    parser.add_argument('--sampling_rate', type=int, default=100, choices=[100, 500],
                       help='100 Hz (faster) or 500 Hz (higher resolution)')
    parser.add_argument('--label_type', default='diagnostic_superclass',
                       choices=['diagnostic_superclass', 'diagnostic_class', 'all'],
                       help='diagnostic_superclass=5 classes, diagnostic_class=~23, all=71')
    parser.add_argument('--max_records', type=int, default=None, 
                       help='Limit records (for testing)')
    parser.add_argument('--n_workers', type=int, default=1, 
                       help='Parallel workers (use 1 if errors)')
    parser.add_argument('--save_per_record', action='store_true', 
                       help='Save each record separately')
    
    args = parser.parse_args()
    
    preprocessor = PTBXLPreprocessorSOTA(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        sampling_rate=args.sampling_rate,
        label_type=args.label_type
    )
    
    preprocessor.run(
        max_records=args.max_records,
        n_workers=args.n_workers,
        save_per_record=args.save_per_record
    )


if __name__ == "__main__":
    main()
