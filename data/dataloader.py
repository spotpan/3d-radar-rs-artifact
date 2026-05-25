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
                 radar_height_layers: List[int] = [0, 1, 2, 3, 4, 5],
                 spatial_size: Tuple[int, int] = (700, 900),
                 use_valid_only: bool = True):
        self.data_paths = data_paths
        self.radar_height_layers = radar_height_layers
        self.spatial_size = spatial_size
        self.use_valid_only = use_valid_only

        self.files = [None] * len(data_paths)
        self.indices = []

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

                    radar_shape = grp['radar'].shape
                    rain_shape = grp['rain'].shape
                    time_shape = grp['time'].shape

                    assert radar_shape[0] == rain_shape[0] == time_shape[0], \
                        f"Mismatched lengths in {file_path}"
                    assert radar_shape[1] >= max(self.radar_height_layers) + 1, \
                        f"Radar only has {radar_shape[1]} layers, but need up to {max(self.radar_height_layers)}"
                    assert radar_shape[2:] == self.spatial_size, \
                        f"Radar spatial size {radar_shape[2:]} != expected {self.spatial_size}"

                    radar_valid = grp['radar_valid'][:]
                    rain_valid = grp['rain_valid'][:]

                    times = grp['time'][:]
                    time_strs = [t.decode('utf-8').strip() for t in times]

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
        if self.files[file_idx] is None:
            self.files[file_idx] = h5py.File(self.data_paths[file_idx], 'r')
        return self.files[file_idx]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        file_idx, sample_idx, time_str = self.indices[idx]
        f = self._get_file_handle(file_idx)
        grp = f['radar-rain']

        radar_full = grp['radar'][sample_idx]
        radar = radar_full[self.radar_height_layers]

        rain = grp['rain'][sample_idx]

        radar_tensor = torch.from_numpy(radar.astype(np.float32))
        rain_tensor = torch.from_numpy(rain.astype(np.float32)).unsqueeze(0)

        return {
            'radar': radar_tensor,
            'rain': rain_tensor,
            'time': time_str,
            'file_idx': file_idx,
            'sample_idx': sample_idx
        }

    def close(self):
        for i, f in enumerate(self.files):
            if f is not None:
                f.close()
                self.files[i] = None

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
        item = super().__getitem__(idx)
        return {
            'radar': item['radar'],
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
                 target_minutes: List[int] = [0, 30],
                 history_frames: int = 6,
                 frame_interval: int = 12):

        super().__init__(data_paths, radar_height_layers, spatial_size, use_valid_only=True)

        self.target_minutes = target_minutes
        self.history_frames = history_frames
        self.frame_interval = frame_interval

        self._build_time_index()
        self._filter_valid_sequences()

    def _build_time_index(self):
        self.time_to_index = {}
        for idx, (file_idx, sample_idx, time_str) in enumerate(self.indices):
            self.time_to_index[time_str] = (file_idx, sample_idx, idx)

    def _parse_time(self, time_str: str) -> datetime:
        try:
            return datetime.strptime(time_str, '%Y%m%d%H%M')
        except ValueError:
            if len(time_str) == 14:
                return datetime.strptime(time_str, '%Y%m%d%H%M%S')
            else:
                raise ValueError(f"Cannot parse time string: {time_str}")

    def _get_time_string(self, dt: datetime) -> str:
        return dt.strftime('%Y%m%d%H%M')

    def _filter_valid_sequences(self):
        valid_indices = []

        for idx, (file_idx, sample_idx, time_str) in enumerate(self.indices):
            try:
                target_time = self._parse_time(time_str)

                if target_time.minute not in self.target_minutes:
                    continue

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

        self.original_indices = self.indices.copy()
        self.indices = [self.original_indices[i] for i in valid_indices]

        print(f"Filtered to {len(self.indices)} valid sequences (from {len(self.original_indices)} total samples)")

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        file_idx, sample_idx, target_time_str = self.indices[idx]
        target_time = self._parse_time(target_time_str)

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

        radar_sequence = np.stack(radar_frames, axis=0)

        f_target = self._get_file_handle(file_idx)
        grp_target = f_target['radar-rain']
        rain = grp_target['rain'][sample_idx]

        radar_tensor = torch.from_numpy(radar_sequence.astype(np.float32))
        rain_tensor = torch.from_numpy(rain.astype(np.float32)).unsqueeze(0)

        return {
            'radar_sequence': radar_tensor,
            'rain': rain_tensor,
            'target_time': target_time_str,
            'history_times': [self._get_time_string(target_time + timedelta(minutes=-i*self.frame_interval))
                            for i in range(self.history_frames)]
        }


class FinetuneDatasetGuangdong(Dataset):
    """Fine-tuning dataset adapted for Guangdong radar data.

    In Guangdong data, valid rain observations are only at minutes 00 and 30.
    However, radar data is available at 6-minute resolution for ALL timesteps.
    This dataset builds the time index from ALL radar samples (ignoring rain_valid)
    so that history frames at any 6-minute interval are accessible, while only
    targeting times where rain observations are valid.

    This allows using the original 6 frames x 12-minute interval configuration.
    """

    def __init__(self,
                 data_paths: List[str],
                 radar_height_layers: List[int] = [0, 1, 2, 3, 4, 5],
                 spatial_size: Tuple[int, int] = (700, 900),
                 target_minutes: List[int] = [0, 30],
                 history_frames: int = 6,
                 frame_interval: int = 12):
        self.data_paths = data_paths
        self.radar_height_layers = radar_height_layers
        self.spatial_size = spatial_size
        self.target_minutes = target_minutes
        self.history_frames = history_frames
        self.frame_interval = frame_interval

        self.files = [None] * len(data_paths)

        # Load ALL radar indices + separately track rain validity
        self._load_all_indices()

    def _load_all_indices(self):
        """Load all radar samples and build time index + valid rain set."""
        self.all_indices = []        # (file_idx, sample_idx, time_str) for ALL samples
        self.time_to_index = {}      # time_str -> (file_idx, sample_idx, idx)
        self.valid_rain_times = set()  # set of time_str with valid rain
        self.valid_indices = []      # final indices for valid sequences

        for file_idx, file_path in enumerate(self.data_paths):
            if not os.path.exists(file_path):
                warnings.warn(f"File not found: {file_path}")
                continue

            try:
                with h5py.File(file_path, 'r') as f:
                    grp = f['radar-rain']

                    radar_shape = grp['radar'].shape
                    rain_shape = grp['rain'].shape
                    time_shape = grp['time'].shape

                    assert radar_shape[0] == rain_shape[0] == time_shape[0], \
                        f"Mismatched lengths in {file_path}"
                    assert radar_shape[1] >= max(self.radar_height_layers) + 1, \
                        f"Radar only has {radar_shape[1]} layers"
                    assert radar_shape[2:] == self.spatial_size, \
                        f"Radar spatial size {radar_shape[2:]} != expected {self.spatial_size}"

                    times = grp['time'][:]
                    time_strs = [t.decode('utf-8').strip() for t in times]

                    radar_valid = grp['radar_valid'][:]
                    rain_valid = grp['rain_valid'][:]

                    # Build time index from all samples (radar data is always available)
                    n_samples = radar_shape[0]
                    for sample_idx in range(n_samples):
                        ts = time_strs[sample_idx]
                        self.all_indices.append((file_idx, sample_idx, ts))
                        self.time_to_index[ts] = (file_idx, sample_idx, sample_idx)
                        if radar_valid[sample_idx] and rain_valid[sample_idx]:
                            self.valid_rain_times.add(ts)

                print(f"Loaded {n_samples} samples from {file_path}")

            except Exception as e:
                warnings.warn(f"Error loading {file_path}: {e}")

        # Now filter for valid sequences
        self._filter_sequences()

        print(f"Total all-radar indices: {len(self.all_indices)}")
        print(f"Total valid rain times: {len(self.valid_rain_times)}")
        print(f"Total valid sequences: {len(self.valid_indices)}")

    def _filter_sequences(self):
        """Filter to target times with valid rain and complete radar history."""
        for idx, (file_idx, sample_idx, time_str) in enumerate(self.all_indices):
            try:
                target_time = datetime.strptime(time_str, '%Y%m%d%H%M')

                # Must be a target minute
                if target_time.minute not in self.target_minutes:
                    continue

                # Must have valid rain at target time
                if time_str not in self.valid_rain_times:
                    continue

                # Check complete radar history
                valid_sequence = True
                for i in range(self.history_frames):
                    history_minutes = -i * self.frame_interval
                    history_time = target_time + timedelta(minutes=history_minutes)
                    history_str = history_time.strftime('%Y%m%d%H%M')

                    if history_str not in self.time_to_index:
                        valid_sequence = False
                        break

                if valid_sequence:
                    self.valid_indices.append(idx)

            except Exception as e:
                warnings.warn(f"Error processing sample {time_str}: {e}")
                continue

    def _get_file_handle(self, file_idx: int) -> h5py.File:
        if self.files[file_idx] is None:
            self.files[file_idx] = h5py.File(self.data_paths[file_idx], 'r')
        return self.files[file_idx]

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        all_idx = self.valid_indices[idx]
        file_idx, sample_idx, target_time_str = self.all_indices[all_idx]
        target_time = datetime.strptime(target_time_str, '%Y%m%d%H%M')

        # Collect radar frames from the time index
        radar_frames = []
        for i in range(self.history_frames):
            history_minutes = -i * self.frame_interval
            history_time = target_time + timedelta(minutes=history_minutes)
            history_str = history_time.strftime('%Y%m%d%H%M')

            h_file_idx, h_sample_idx, _ = self.time_to_index[history_str]
            f = self._get_file_handle(h_file_idx)
            grp = f['radar-rain']

            radar_full = grp['radar'][h_sample_idx]
            radar = radar_full[self.radar_height_layers]
            radar_frames.append(radar)

        radar_sequence = np.stack(radar_frames, axis=0)

        # Get target precipitation
        f_target = self._get_file_handle(file_idx)
        grp_target = f_target['radar-rain']
        rain = grp_target['rain'][sample_idx]

        radar_tensor = torch.from_numpy(radar_sequence.astype(np.float32))
        rain_tensor = torch.from_numpy(rain.astype(np.float32)).unsqueeze(0)

        return {
            'radar_sequence': radar_tensor,
            'rain': rain_tensor,
            'target_time': target_time_str,
            'history_times': [
                (target_time + timedelta(minutes=-i * self.frame_interval)).strftime('%Y%m%d%H%M')
                for i in range(self.history_frames)
            ]
        }

    def compute_sample_weights(
        self,
        thresholds=(0.1, 1.0, 3.5, 7.5),
        area_rules=None,
        max_weight: float = 4.0,
        verbose: bool = True,
    ):
        """Compute sample weights from target rain structural area.

        Each dataset item corresponds to a target rain frame. This method assigns
        larger weights to samples containing spatially meaningful rain structures,
        instead of only using the maximum rain value.

        Default area_rules:
            rain >= 0.1 area >= 0.005: +0.5
            rain >= 1.0 area >= 0.002: +0.7
            rain >= 3.5 area >= 0.001: +0.8
            rain >= 7.5 area >= 0.002: +1.0
        """
        if area_rules is None:
            area_rules = [
                (0.1, 0.005, 0.5),
                (1.0, 0.002, 0.7),
                (3.5, 0.001, 0.8),
                (7.5, 0.002, 1.0),
            ]

        sample_weights = []
        record_types = []
        area_stats = []

        type_counts = np.zeros(len(thresholds) + 1, dtype=np.int64)
        rule_hits = np.zeros(len(area_rules), dtype=np.int64)

        for ds_idx, all_idx in enumerate(self.valid_indices):
            file_idx, sample_idx, target_time_str = self.all_indices[all_idx]

            f_target = self._get_file_handle(file_idx)
            grp_target = f_target['radar-rain']
            rain = grp_target['rain'][sample_idx].astype(np.float32)

            finite = np.isfinite(rain)
            vals = rain[finite]

            if vals.size == 0:
                max_rain = 0.0
                weight = 1.0
                areas = {float(thr): 0.0 for thr, _, _ in area_rules}
            else:
                max_rain = float(vals.max())
                weight = 1.0
                areas = {}

                for rule_idx, (rain_thr, area_cutoff, bonus) in enumerate(area_rules):
                    area_ratio = float((vals >= rain_thr).sum()) / max(vals.size, 1)
                    areas[float(rain_thr)] = area_ratio
                    if area_ratio >= area_cutoff:
                        weight += float(bonus)
                        rule_hits[rule_idx] += 1

            weight = min(float(weight), float(max_weight))
            rec_type = int(np.digitize(max_rain, thresholds, right=False))

            sample_weights.append(weight)
            record_types.append(rec_type)
            area_stats.append(areas)
            type_counts[rec_type] += 1

        self.sample_weights = np.asarray(sample_weights, dtype=np.float32)
        self.record_types = np.asarray(record_types, dtype=np.int64)
        self.area_stats = area_stats

        if verbose:
            print("\nSample weights computed:")
            print(f"  num samples: {len(self.sample_weights)}")
            print(f"  min / max / mean weight: "
                  f"{self.sample_weights.min():.4f} / "
                  f"{self.sample_weights.max():.4f} / "
                  f"{self.sample_weights.mean():.4f}")

            print("  record type counts by max rain:")
            labels = []
            labels.append(f"type 0: max < {thresholds[0]}")
            for i in range(1, len(thresholds)):
                labels.append(f"type {i}: {thresholds[i-1]} <= max < {thresholds[i]}")
            labels.append(f"type {len(thresholds)}: max >= {thresholds[-1]}")

            total = max(len(self.sample_weights), 1)
            for label, count in zip(labels, type_counts):
                print(f"    {label:32s}: {count:8d} ({count / total:.4f})")

            print("  area rule hits:")
            for (rain_thr, area_cutoff, bonus), hit in zip(area_rules, rule_hits):
                print(f"    rain >= {rain_thr}, area >= {area_cutoff}: "
                      f"{hit:8d} ({hit / total:.4f}), bonus={bonus}")

        return self.sample_weights

    def close(self):
        for i, f in enumerate(self.files):
            if f is not None:
                f.close()
                self.files[i] = None

    def __del__(self):
        self.close()


# For testing
if __name__ == "__main__":
    data_path = "/path/to/radar_station_dataset/time_radar_rain_2022.h5"

    print("=== Testing FinetuneDatasetGuangdong ===")
    ds = FinetuneDatasetGuangdong(
        data_paths=[data_path],
        radar_height_layers=[0, 1, 2, 3, 4, 5],
        spatial_size=(700, 900),
        target_minutes=[0, 30],
        history_frames=6,
        frame_interval=12,
    )
    if len(ds) > 0:
        sample = ds[0]
        print(f"Radar sequence shape: {sample['radar_sequence'].shape}")
        print(f"Rain shape: {sample['rain'].shape}")
        print(f"Target time: {sample['target_time']}")

    ds.close()
