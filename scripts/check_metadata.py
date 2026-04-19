import h5py
import numpy as np

file_path = "/mnt/md1/hxc/guangdong/train_select/time_radar_rain_2022.h5"

with h5py.File(file_path, 'r') as f:
    grp = f['radar-rain']

    # Check datasets
    for name, obj in grp.items():
        if isinstance(obj, h5py.Dataset):
            print(f"{name}: shape={obj.shape}, dtype={obj.dtype}, chunks={obj.chunks}")
            # Read a small sample for non-large datasets
            if obj.shape[0] < 100000:  # small enough
                if obj.dtype.kind == 'S':  # string
                    sample = obj[:5]
                    print(f"  Sample: {[s.decode('utf-8').strip() for s in sample]}")
                else:
                    print(f"  Min/Max: {obj[:].min()}, {obj[:].max()}")
        else:
            print(f"{name}: Group with {len(obj)} items")

    # Specifically check time format
    times = grp['time'][:10]
    time_strs = [t.decode('utf-8').strip() for t in times]
    print(f"\nFirst 10 times: {time_strs}")

    # Check if consistent length
    lengths = set(len(t) for t in time_strs)
    print(f"Time string lengths: {lengths}")

    # Radar channels info
    radar = grp['radar']
    print(f"\nRadar has {radar.shape[1]} channels (height layers)")

    # Check a small sample from first channel
    radar_sample = radar[:10, 0, 350:360, 450:460]  # tiny spatial slice
    print(f"Radar sample (first channel, small spatial slice):")
    print(radar_sample)