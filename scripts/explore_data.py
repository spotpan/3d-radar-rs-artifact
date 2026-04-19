import h5py
import numpy as np
import sys
from datetime import datetime

def explore_file(file_path):
    print(f"Exploring {file_path}")
    with h5py.File(file_path, 'r') as f:
        grp = f['radar-rain']

        # Time data
        times = grp['time'][:]
        print(f"Time shape: {times.shape}, dtype: {times.dtype}")
        print(f"First 5 times: {times[:5]}")
        print(f"Last 5 times: {times[-5:]}")

        # Convert bytes to string
        time_strs = [t.decode('utf-8').strip() for t in times[:10]]
        print(f"First 10 time strings: {time_strs}")

        # Radar data
        radar = grp['radar']
        print(f"Radar shape: {radar.shape}, dtype: {radar.dtype}")
        print(f"Radar min/max: {radar[:].min()}, {radar[:].max()}")

        # Check each channel
        for i in range(radar.shape[1]):
            chan_data = radar[:100, i, :, :]  # sample first 100 timesteps
            print(f"Channel {i}: min={chan_data.min()}, max={chan_data.max()}, mean={chan_data.mean():.2f}")

        # Rain data
        rain = grp['rain']
        print(f"Rain shape: {rain.shape}, dtype: {rain.dtype}")
        rain_sample = rain[:100]
        print(f"Rain min/max: {rain_sample.min()}, {rain_sample.max()}")
        print(f"Rain non-zero fraction: {(rain_sample > 0).sum() / rain_sample.size:.3f}")

        # Valid masks
        radar_valid = grp['radar_valid'][:]
        rain_valid = grp['rain_valid'][:]
        print(f"Radar valid: {radar_valid.sum()}/{len(radar_valid)} ({radar_valid.sum()/len(radar_valid):.2%})")
        print(f"Rain valid: {rain_valid.sum()}/{len(rain_valid)} ({rain_valid.sum()/len(rain_valid):.2%})")

        # Check alignment
        both_valid = radar_valid & rain_valid
        print(f"Both valid: {both_valid.sum()}/{len(both_valid)} ({both_valid.sum()/len(both_valid):.2%})")

        # Find time pattern
        if len(time_strs) > 0:
            # Try to parse time format
            sample = time_strs[0]
            print(f"Sample time string: '{sample}'")
            # Check if it's like '202201010000' (YYYYMMDDHHMM)
            if len(sample) == 12:
                print("Time format appears to be YYYYMMDDHHMM")
                try:
                    dt = datetime.strptime(sample, '%Y%m%d%H%M')
                    print(f"Parsed as: {dt}")
                except:
                    print("Failed to parse as YYYYMMDDHHMM")

if __name__ == "__main__":
    file_path = "/mnt/md1/hxc/guangdong/train_select/time_radar_rain_2022.h5"
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    explore_file(file_path)