import os
import librosa
import numpy as np
import matplotlib.pyplot as plt

base_dir = os.path.dirname(__file__)

# Update this with the exact path of your selected audio file
sample_file = r"GROUP_06\G06_D04_M_48_Furious_C3.wav"
file_name = os.path.basename(sample_file) # Extract the file name automatically

audio, sr = librosa.load(sample_file, sr=22050)

# Extract a single frame (from the middle of the audio)
window_size = int(sr * 0.025)  # 25ms window
mid = len(audio) // 2
frame = audio[mid:mid + window_size]

# --- AUTOCORRELATION METHOD ---
corr = np.correlate(frame, frame, mode='full')
corr = corr[len(corr)//2:]
corr = corr / corr[0]  # Normalize the correlation array

# Limit search to human pitch range (50 Hz - 500 Hz)
lag_min = int(sr / 500)
lag_max = int(sr / 50)
peak_lag = np.argmax(corr[lag_min:lag_max]) + lag_min
f0_autocorr = sr / peak_lag

# --- FFT METHOD ---
N = len(frame)
fft_magnitude = np.abs(np.fft.rfft(frame))
freqs = np.fft.rfftfreq(N, d=1/sr)

# Filter frequencies between 50 Hz and 500 Hz
fft_range = (freqs >= 50) & (freqs <= 500)
f0_fft = freqs[fft_range][np.argmax(fft_magnitude[fft_range])]

# --- PLOTTING ---
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# Autocorrelation Plot
lags = np.arange(len(corr[:lag_max+50]))
axes[0].plot(sr / np.maximum(lags[lag_min:], 1), corr[lag_min:lag_max+50], color='steelblue')
axes[0].axvline(f0_autocorr, color='red', linestyle='--', label=f'F0 = {f0_autocorr:.1f} Hz')
axes[0].set_title('Autocorrelation Method')
axes[0].set_xlabel('Frequency (Hz)')
axes[0].set_ylabel('Autocorrelation')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# FFT Plot
axes[1].plot(freqs[fft_range], fft_magnitude[fft_range], color='darkorange')
axes[1].axvline(f0_fft, color='red', linestyle='--', label=f'F0 = {f0_fft:.1f} Hz')
axes[1].set_title('FFT Method')
axes[1].set_xlabel('Frequency (Hz)')
axes[1].set_ylabel('Magnitude')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle(f'Autocorrelation vs. FFT Comparison\nFile: {file_name}\nAutocorrelation: {f0_autocorr:.1f} Hz  |  FFT: {f0_fft:.1f} Hz', 
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("autocorr_vs_fft.png", dpi=150, bbox_inches='tight')
plt.show()

# --- CONSOLE OUTPUTS ---
print(f"Autocorrelation F0 : {f0_autocorr:.1f} Hz")
print(f"FFT F0           : {f0_fft:.1f} Hz")
print("autocorr_vs_fft.png successfully saved!")