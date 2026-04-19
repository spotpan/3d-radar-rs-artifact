import h5py
import sys

file_path = sys.argv[1] if len(sys.argv) > 1 else "/mnt/md1/hxc/guangdong/train_select/time_radar_rain_2022.h5"

try:
    with h5py.File(file_path, 'r') as f:
        print(f"File: {file_path}")
        print("Keys:", list(f.keys()))
        for key in f.keys():
            print(f"\n--- {key} ---")
            if isinstance(f[key], h5py.Dataset):
                print(f"  Shape: {f[key].shape}")
                print(f"  Dtype: {f[key].dtype}")
                # Show a small sample
                if f[key].ndim > 0:
                    print(f"  First few elements (flattened): {f[key][:].flatten()[:5]}")
            else:
                print(f"  Group with keys: {list(f[key].keys())}")
                for subkey in f[key].keys():
                    if isinstance(f[key][subkey], h5py.Dataset):
                        print(f"    {subkey}: shape {f[key][subkey].shape}, dtype {f[key][subkey].dtype}")
except Exception as e:
    print(f"Error: {e}")