import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from datetime import datetime, timedelta
import os
from typing import Tuple, List, Optional, Dict, Any
import warnings


class RadarRainDataset(Dataset):
    """Base dataset for radar and rain data."""

    def __init__(self,
                 data_paths: List[str],
                 radar_height_layers: List[int] = [0, 1, 2, 3, 4, 5],  # Use first 6 layers (0-indexed)
                 spatial_size: Tuple[int, int] = (700, 900),
                 use_valid_only: bool = True):
        """
        Args:
            data_paths: List of HDF5 file paths
            radar_height_layers: Which height layers to use (0-indexed)
            spatial_size: Expected spatial size (H, W)
            use_valid_only: Whether to only use samples where both radar and rain are valid
        """
        self.data_paths = data_paths
        self.radar_height_layers = radar_height_layers
        self.spatial_size = spatial_size
        self.use_valid_only = use_valid_only

        # Store file handles
        self.files = []
        # Store indices as (file_idx, sample_idx)
        self.indices = []

        # Parse all files
        self._load_indices()

    def _load_indices(self):
        """Load indices from all files."""
        self.indices = []

        for file_idx, file_path in enumerate(self.data_paths):
            if not os.path.exists(file_path):
                warnings.warn(f"File not found: {file_path}")
                continue

            try:
                with h5py.File(file_path, 'r') as f:
                    grp = f['radar-rain']

                    # Check shapes
                    radar_shape = grp['radar'].shape
                    rain_shape = grp['rain'].shape
                    time_shape = grp['time'].shape

                    assert radar_shape[0] == rain_shape[0] == time_shape[0], \
                        f"Mismatched lengths in {file_path}"
                    assert radar_shape[1] >= max(self.radar_height_layers) + 1, \
                        f"Radar only has {radar_shape[1]} layers, but need up to {max(self.radar_height_layers)}"
                    assert radar_shape[2:] == self.spatial_size, \
                        f"Radar spatial size {radar_shape[2:]} != expected {self.spatial_size}"

                    # Get valid masks
                    radar_valid = grp['radar_valid'][:]
                    rain_valid = grp['rain_valid'][:]

                    # Parse times
                    times = grp['time'][:]
                    time_strs = [t.decode('utf-8').strip() for t in times]

                    # Create indices
                    n_samples = radar_shape[0]
                    for sample_idx in range(n_samples):
                        if self.use_valid_only:
                            if not (radar_valid[sample_idx] and rain_valid[sample_idx]):
                                continue
                        self.indices.append((file_idx, sample_idx, time_strs[sample_idx]))

                print(f"Loaded {len([i for i in self.indices if i[0] == file_idx])} samples from {file_path}")

            except Exception as e:
                warnings.warn(f"Error loading {file_path}: {e}")

        print(f"Total samples: {len(self.indices)}")

    def _get_file_handle(self, file_idx: int) -> h5py.File:
        """Get or create file handle."""
        if len(self.files) <= file_idx:
            # Open new file
            self.files.append(h5py.File(self.data_paths[file_idx], 'r'))
        return self.files[file_idx]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        file_idx, sample_idx, time_str = self.indices[idx]
        f = self._get_file_handle(file_idx)
        grp = f['radar-rain']

        # Get radar data (N, 11, H, W) -> select height layers
        radar_full = grp['radar'][sample_idx]  # (11, H, W)
        radar = radar_full[self.radar_height_layers]  # (6, H, W)

        # Get rain data
        rain = grp['rain'][sample_idx]  # (H, W)

        # Convert to torch tensors
        radar_tensor = torch.from_numpy(radar.astype(np.float32))
        rain_tensor = torch.from_numpy(rain.astype(np.float32)).unsqueeze(0)  # (1, H, W)

        return {
            'radar': radar_tensor,  # (6, H, W)
            'rain': rain_tensor,    # (1, H, W)
            'time': time_str,
            'file_idx': file_idx,
            'sample_idx': sample_idx
        }

    def close(self):
        """Close all open file handles."""
        for f in self.files:
            f.close()
        self.files = []

    def __del__(self):
        self.close()


class PretrainDataset(RadarRainDataset):
    """Dataset for 3DMAE pretraining (radar only, all timesteps)."""

    def __init__(self,
                 data_paths: List[str],
                 radar_height_layers: List[int] = [0, 1, 2, 3, 4, 5],
                 spatial_size: Tuple[int, int] = (700, 900)):
        super().__init__(data_paths, radar_height_layers, spatial_size, use_valid_only=False)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Return only radar data for pretraining."""
        item = super().__getitem__(idx)
        return {
            'radar': item['radar'],  # (6, H, W)
            'time': item['time']
        }


class FinetuneDataset(RadarRainDataset):
    """Dataset for precipitation estimation fine-tuning.

    For each precipitation target time (hourly at 00 and 30 minutes),
    we need 6 frames of radar data at 12-minute intervals.
    """

    def __init__(self,
                 data_paths: List[str],
                 radar_height_layers: List[int] = [0, 1, 2, 3, 4, 5],
                 spatial_size: Tuple[int, int] = (700, 900),
                 target_minutes: List[int] = [0, 30],  # Precipitation target minutes
                 history_frames: int = 6,  # Number of radar frames
                 frame_interval: int = 12):  # Minutes between frames

        super().__init__(data_paths, radar_height_layers, spatial_size, use_valid_only=True)

        self.target_minutes = target_minutes
        self.history_frames = history_frames
        self.frame_interval = frame_interval

        # Build time index for efficient lookup
        self._build_time_index()

        # Filter indices to only include valid sequences
        self._filter_valid_sequences()

    def _build_time_index(self):
        """Build index mapping from time string to (file_idx, sample_idx)."""
        self.time_to_index = {}
        for idx, (file_idx, sample_idx, time_str) in enumerate(self.indices):
            self.time_to_index[time_str] = (file_idx, sample_idx, idx)

    def _parse_time(self, time_str: str) -> datetime:
        """Parse time string to datetime object.

        Expected format: YYYYMMDDHHMM (12 characters)
        """
        try:
            return datetime.strptime(time_str, '%Y%m%d%H%M')
        except ValueError:
            # Try other possible formats
            if len(time_str) == 14:  # YYYYMMDDHHMMSS
                return datetime.strptime(time_str, '%Y%m%d%H%M%S')
            else:
                raise ValueError(f"Cannot parse time string: {time_str}")

    def _get_time_string(self, dt: datetime) -> str:
        """Convert datetime back to time string."""
        return dt.strftime('%Y%m%d%H%M')

    def _filter_valid_sequences(self):
        """Filter indices to only include samples that have complete history sequences."""
        valid_indices = []

        for idx, (file_idx, sample_idx, time_str) in enumerate(self.indices):
            try:
                target_time = self._parse_time(time_str)

                # Check if this is a target minute
                if target_time.minute not in self.target_minutes:
                    continue

                # Check if we have all history frames
                valid_sequence = True
                for i in range(self.history_frames):
                    history_minutes = -i * self.frame_interval
                    history_time = target_time + timedelta(minutes=history_minutes)
                    history_str = self._get_time_string(history_time)

                    if history_str not in self.time_to_index:
                        valid_sequence = False
                        break

                if valid_sequence:
                    valid_indices.append(idx)

            except Exception as e:
                warnings.warn(f"Error processing sample {time_str}: {e}")
                continue

        # Update indices
        self.original_indices = self.indices.copy()
        self.indices = [self.original_indices[i] for i in valid_indices]

        print(f"Filtered to {len(self.indices)} valid sequences (from {len(self.original_indices)} total samples)")

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Return sequence of radar frames and target precipitation."""
        file_idx, sample_idx, target_time_str = self.indices[idx]
        target_time = self._parse_time(target_time_str)

        # Collect radar frames
        radar_frames = []
        for i in range(self.history_frames):
            history_minutes = -i * self.frame_interval
            history_time = target_time + timedelta(minutes=history_minutes)
            history_str = self._get_time_string(history_time)

            if history_str not in self.time_to_index:
                raise ValueError(f"Missing history frame at {history_str}")

            h_file_idx, h_sample_idx, _ = self.time_to_index[history_str]
            f = self._get_file_handle(h_file_idx)
            grp = f['radar-rain']

            radar_full = grp['radar'][h_sample_idx]
            radar = radar_full[self.radar_height_layers]
            radar_frames.append(radar)

        # Stack frames
        radar_sequence = np.stack(radar_frames, axis=0)  # (T, 6, H, W)

        # Get target precipitation
        f_target = self._get_file_handle(file_idx)
        grp_target = f_target['radar-rain']
        rain = grp_target['rain'][sample_idx]  # (H, W)

        # Convert to torch tensors
        radar_tensor = torch.from_numpy(radar_sequence.astype(np.float32))
        rain_tensor = torch.from_numpy(rain.astype(np.float32)).unsqueeze(0)  # (1, H, W)

        return {
            'radar_sequence': radar_tensor,  # (T, 6, H, W)
            'rain': rain_tensor,             # (1, H, W)
            'target_time': target_time_str,
            'history_times': [self._get_time_string(target_time + timedelta(minutes=-i*self.frame_interval))
                            for i in range(self.history_frames)]
        }


# For testing
if __name__ == "__main__":
    from datetime import timedelta

    # Test with small subset
    data_path = "/mnt/md1/hxc/guangdong/train_select/time_radar_rain_2022.h5"

    print("Testing RadarRainDataset...")
    dataset = RadarRainDataset([data_path], use_valid_only=True)
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Radar shape: {sample['radar'].shape}")
        print(f"Rain shape: {sample['rain'].shape}")
        print(f"Time: {sample['time']}")

    print("\nTesting PretrainDataset...")
    pretrain_dataset = PretrainDataset([data_path])
    if len(pretrain_dataset) > 0:
        sample = pretrain_dataset[0]
        print(f"Radar shape: {sample['radar'].shape}")

    print("\nTesting FinetuneDataset...")
    finetune_dataset = FinetuneDataset([data_path])
    if len(finetune_dataset) > 0:
        sample = finetune_dataset[0]
        print(f"Radar sequence shape: {sample['radar_sequence'].shape}")
        print(f"Rain shape: {sample['rain'].shape}")
        print(f"Target time: {sample['target_time']}")
        print(f"History times: {sample['history_times'][:3]}...")

    # Clean up
    dataset.close()