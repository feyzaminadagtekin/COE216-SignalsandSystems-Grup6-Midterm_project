import numpy as np
import pickle

with open("results_final.pkl", "rb") as f:
    results = pickle.load(f)
print(f"Loaded: {len(results)} files")

def compute_f0_autocorrelation(frames, voiced_mask, sr, hop_size, f0_min=50, f0_max=500):
    voiced_frames = frames[:, voiced_mask]
    if voiced_frames.shape[1] == 0:
        return None
    f0_values = []
    for i in range(voiced_frames.shape[1]):
        frame = voiced_frames[:, i]
        corr = np.correlate(frame, frame, mode='full')
        corr = corr[len(corr)//2:]
        lag_min = max(1, int(sr / f0_max))
        lag_max = min(int(sr / f0_min), len(corr) - 1)
        search_region = corr[lag_min:lag_max]
        if len(search_region) == 0:
            continue
        peak_lag = np.argmax(search_region) + lag_min
        f0_values.append(sr / peak_lag)
    return np.mean(f0_values) if f0_values else None

print("Calculating F0...")
for i, r in enumerate(results):
    r['f0'] = compute_f0_autocorrelation(r['frames'], r['voiced_mask'], r['sr'], r['hop_size'])
    if i % 50 == 0:
        print(f"{i}/{len(results)} processed...")

f0_by_class = {'male': [], 'female': [], 'child': []}
for r in results:
    if r['f0'] is not None:
        f0_by_class[r['actual_class']].append(r['f0'])

print("\nF0 Statistics:")
for cls, f0_list in f0_by_class.items():
    print(f"  {cls:8s} → Mean: {np.mean(f0_list):.1f} Hz  |  Std: {np.std(f0_list):.1f} Hz  |  N: {len(f0_list)}")

with open("results_f0.pkl", "wb") as f:
    pickle.dump(results, f)
print("\nresults_f0.pkl saved!")