import pandas as pd
import numpy as np
import librosa
import os
import pickle
from collections import Counter

df = pd.read_excel("master_metadata.xlsx")
df = df[df['audio_file_present'] == True].reset_index(drop=True)

def find_actual_path(relative_path):
    if os.path.exists(relative_path):
        return relative_path
    parts = relative_path.replace("\\", "/").split("/")
    group_folder = parts[-2]
    file_name = parts[-1]
    candidates = [
        group_folder,
        group_folder.replace("GROUP_", "GRUP_"),
        group_folder.replace("GRUP_", "GROUP_"),
        group_folder.capitalize(),
    ]
    for candidate in candidates:
        path = os.path.join(candidate, file_name)
        if os.path.exists(path):
            return path
    return None

df['real_path'] = df['audio_relative_path'].apply(find_actual_path)
print(f"[STEP 1] Files found: {df['real_path'].notna().sum()}")

def analyze_file(file_path, sr=22050, window_ms=25, hop_ms=10):
    audio, sr = librosa.load(file_path, sr=sr)
    window_size = int(sr * window_ms / 1000)
    hop_size = int(sr * hop_ms / 1000)

    frames = librosa.util.frame(audio, frame_length=window_size, hop_length=hop_size)

    ste = np.sum(frames ** 2, axis=0)

    zcr = librosa.feature.zero_crossing_rate(
        audio,
        frame_length=window_size,
        hop_length=hop_size
    )[0]

    min_len = min(len(ste), len(zcr))
    ste = ste[:min_len]
    zcr = zcr[:min_len]
    frames = frames[:, :min_len]

    voiced_mask = (ste > np.mean(ste) * 0.5) & (zcr < np.mean(zcr) * 1.5)

    return frames, ste, zcr, voiced_mask, sr, hop_size

df_found = df[df['real_path'].notna()].reset_index(drop=True)

results = []

for i, row in df_found.iterrows():
    try:
        frames, ste, zcr, voiced_mask, sr, hop_size = analyze_file(row['real_path'])

        results.append({
            'file_name': row['file_name'],
            'actual_class': row['actual_class'],
            'emotion': row['feeling'],
            'voiced_count': int(np.sum(voiced_mask)),
            'total_frames': len(ste),
            'frames': frames,
            'voiced_mask': voiced_mask,
            'ste': ste,
            'zcr': zcr,
            'sr': sr,
            'hop_size': hop_size
        })

    except Exception as e:
        print(f"ERROR - {row['file_name']}: {e}")
        continue

    if i % 50 == 0:
        print(f"[STEP 2] Processed {i}/{len(df_found)} files")

print(f"[STEP 2] Completed. Total processed files: {len(results)}")

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

print("[STEP 3] Computing F0 values...")

for i, r in enumerate(results):

    r['f0'] = compute_f0_autocorrelation(
        r['frames'],
        r['voiced_mask'],
        r['sr'],
        r['hop_size']
    )

    if i % 50 == 0:
        print(f"[STEP 3] Processed {i}/{len(results)} files")

print("[STEP 3] F0 computation completed")

def classify(f0):

    if f0 is None:
        return "unknown"

    if f0 < 200:
        return "male"

    elif f0 < 300:
        return "female"

    else:
        return "child"

correct = 0
total = 0
errors = []

confusion = {
    'male': Counter(),
    'female': Counter(),
    'child': Counter()
}

for r in results:

    if r['f0'] is None:
        continue

    actual = r['actual_class']
    predicted = classify(r['f0'])

    r['predicted_class'] = predicted

    confusion[actual][predicted] += 1

    if actual == predicted:
        correct += 1
    else:
        errors.append({
            'file': r['file_name'],
            'actual': actual,
            'predicted': predicted,
            'f0': round(r['f0'], 1)
        })

    total += 1

accuracy = correct / total * 100

print(f"\n[STEP 4] Overall Accuracy: %{accuracy:.1f} ({correct}/{total})")

print("\n[STEP 4] Class Accuracy:")

for cls in ['male', 'female', 'child']:

    n = sum(confusion[cls].values())

    acc = confusion[cls][cls] / n * 100 if n > 0 else 0

    print(f"{cls:8s} → %{acc:.1f} ({confusion[cls][cls]}/{n})")

print("\n[STEP 4] Confusion Matrix")

print(f"{'':10s} {'male':>8s} {'female':>8s} {'child':>8s}")

for cls in ['male', 'female', 'child']:

    row = f"{cls:10s}"

    for pred in ['male', 'female', 'child']:
        row += f"{confusion[cls][pred]:>8d}"

    print(row)

table = []

for cls in ['male', 'female', 'child']:

    f0_list = [
        r['f0'] for r in results
        if r['actual_class'] == cls and r['f0'] is not None
    ]

    correct_count = sum(
        1 for r in results
        if r['actual_class'] == cls and classify(r['f0']) == cls
    )

    total_count = len(f0_list)

    table.append({
        'Class': cls.capitalize(),
        'Sample Count': total_count,
        'Mean F0 (Hz)': round(np.mean(f0_list), 1),
        'Std Dev (Hz)': round(np.std(f0_list), 1),
        'Accuracy (%)': round(correct_count / total_count * 100, 1)
    })

df_table = pd.DataFrame(table)

print("\n[STEP 5] Statistics Table")
print(df_table.to_string(index=False))

df_table.to_excel("statistics_table.xlsx", index=False)

with open("results_final.pkl", "wb") as f:
    pickle.dump(results, f)

print("\n[SAVED] results_final.pkl saved")
print("[SAVED] statistics_table.xlsx saved")